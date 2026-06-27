#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;

struct Detection {
    cv::Rect box;
    int class_id;
    float confidence;
};

enum IntersectionStatus { COMPLIANT, VIOLATION };

class AutonomousPerceptionEngine {
private:
    Ort::Env env;
    unique_ptr<Ort::Session> session_crosswalk;
    unique_ptr<Ort::Session> session_vehicles;
    const cv::Size img_size = cv::Size(640, 640);

public:
    AutonomousPerceptionEngine(wstring crosswalk_path, wstring vehicles_path) {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "EnforcementEngine");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_crosswalk = make_unique<Ort::Session>(env, crosswalk_path.c_str(), session_options);
        session_vehicles = make_unique<Ort::Session>(env, vehicles_path.c_str(), session_options);
        cout << "[INIT] Models loaded. Debugging X-Ray mode enabled." << endl;
    }

    void preprocess(const cv::Mat& frame, vector<float>& input_tensor_values) {
        cv::Mat resized;
        cv::resize(frame, resized, img_size);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
        input_tensor_values.resize(1 * 3 * 640 * 640);
        int channel_elements = 640 * 640;
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 640; ++h) {
                for (int w = 0; w < 640; ++w) {
                    input_tensor_values[c * channel_elements + h * 640 + w] = resized.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }

    void process_image(string image_path) {
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            cerr << "Error: Could not find " << image_path << " - Make sure it is in the same folder as the .exe!" << endl;
            return;
        }
        vector<float> input_tensor;
        cv::resize(frame, frame, img_size);
        preprocess(frame, input_tensor);

        vector<Detection> crosswalk_dets = run_inference(session_crosswalk.get(), input_tensor);
        vector<Detection> vehicle_dets = run_inference(session_vehicles.get(), input_tensor);

        IntersectionStatus status = evaluate_spatial_logic(crosswalk_dets, vehicle_dets);
        render_hud(frame, crosswalk_dets, vehicle_dets, status);
        cv::imshow("Debugging View", frame);
        cv::waitKey(0);
    }

private:
    vector<Detection> run_inference(Ort::Session* session, const vector<float>& input_tensor) {
        vector<Detection> detections;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        vector<int64_t> input_shape = {1, 3, 640, 640};
        Ort::Value input_ort_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_tensor.data()), input_tensor.size(), input_shape.data(), input_shape.size());
        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &input_ort_tensor, 1, output_names, 1);
        
        auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_classes = (int)out_shape[1] - 4; 
        int rows = (int)out_shape[2];            
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        vector<cv::Rect> boxes;
        vector<float> confidences;
        vector<int> class_ids;

        for (int i = 0; i < rows; ++i) {
            float max_conf = 0.0f;
            int best_class_id = -1;
            for (int c = 0; c < num_classes; ++c) {
                float conf = output_data[(4 + c) * rows + i];
                if (conf > max_conf) { max_conf = conf; best_class_id = c; }
            }
            if (max_conf > 0.25f) { 
                float xc = output_data[0 * rows + i], yc = output_data[1 * rows + i], w = output_data[2 * rows + i], h = output_data[3 * rows + i];
                boxes.push_back(cv::Rect(int(xc - 0.5 * w), int(yc - 0.5 * h), int(w), int(h)));
                confidences.push_back(max_conf);
                class_ids.push_back(best_class_id);
            }
        }
        vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indices);
        for (int idx : indices) detections.push_back({boxes[idx], class_ids[idx], confidences[idx]});
        return detections;
    }

    IntersectionStatus evaluate_spatial_logic(const vector<Detection>& crosswalks, const vector<Detection>& elements) {
        bool red_light_active = false;
        for (const auto& e : elements) {
            if (e.class_id == 17) red_light_active = true; 
        }
        if (!red_light_active) return COMPLIANT;
        for (const auto& cw : crosswalks) {
            for (const auto& e : elements) {
                if ((e.class_id == 11 || e.class_id == 20) && (cw.box & e.box).area() > 0) return VIOLATION;
            }
        }
        return COMPLIANT;
    }

    void render_hud(cv::Mat& frame, const vector<Detection>& crosswalks, const vector<Detection>& elements, IntersectionStatus status) {
        cv::putText(frame, (status == VIOLATION ? "STATUS: VIOLATION DETECTED" : "STATUS: COMPLIANT"), cv::Point(30, 50), cv::FONT_HERSHEY_DUPLEX, 1.2, status == VIOLATION ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0), 2);
        
        for (const auto& e : elements) {
            cv::Scalar color;
            string label_name;

            if (e.class_id == 17) {
                color = cv::Scalar(0, 0, 255);
                label_name = "Red Light";
            } else if (e.class_id == 11 || e.class_id == 20) {
                color = cv::Scalar(255, 0, 0); 
                label_name = "Vehicle";
            } else if (e.class_id == 13) {
                color = cv::Scalar(255, 0, 255); 
                label_name = "Pedestrian";
            } else {
                color = cv::Scalar(0, 255, 255); 
                label_name = "ID: " + to_string(e.class_id); 
            }

            cv::rectangle(frame, e.box, color, 2);
            string tag = label_name + " (" + to_string(int(e.confidence * 100)) + "%)";
            cv::putText(frame, tag, cv::Point(e.box.x, e.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
        
        for (const auto& cw : crosswalks) {
            cv::rectangle(frame, cw.box, cv::Scalar(255, 255, 255), 2);
            string cw_tag = "Crosswalk";
            cv::putText(frame, cw_tag, cv::Point(cw.box.x, cw.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }
    }
};

int main() {
    AutonomousPerceptionEngine system_core(L"model_weights/crosswalk_best.onnx", L"model_weights/C&TL_detector2.onnx");
    system_core.process_image("image2.jpg");
    return 0;
}