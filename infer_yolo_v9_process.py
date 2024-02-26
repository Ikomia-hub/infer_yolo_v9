import copy
from ikomia import core, dataprocess, utils
import torch
import numpy as np
from infer_yolo_v9.ikutils import download_model
from infer_yolo_v9.yolov9.models.common import DetectMultiBackend
from infer_yolo_v9.yolov9.utils.general import non_max_suppression, scale_boxes
from infer_yolo_v9.yolov9.utils.augmentations import letterbox
import os

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYoloV9Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.input_size = 640
        self.use_custom_model = False
        self.model_name = 'yolov9-c'
        self.cuda = torch.cuda.is_available()
        self.conf_thres = 0.25
        self.iou_thres = 0.5
        self.model_weight_file = ""
        self.class_file = ""
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.input_size = int(param_map["input_size"])
        self.use_custom_model = utils.strtobool(param_map["use_custom_model"])
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_thres = float(param_map["iou_thres"])
        self.model_weight_file = param_map["model_weight_file"]
        self.class_file = param_map["class_file"]
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["input_size"] = str(self.input_size)
        param_map['model_name'] = str(self.model_name)
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["iou_thres"] = str(self.iou_thres)
        param_map["cuda"] = str(self.cuda)
        param_map["model_weight_file"] = str(self.model_weight_file)
        param_map["class_file"] = str(self.class_file)

        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloV9(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Add input/output of the algorithm here
        # Create parameters object
        if param is None:
            self.set_param_object(InferYoloV9Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.model = None
        self.weights = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get input :
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = input.get_image()

        # Get parameters :
        param = self.get_param_object()

        if param.update or self.model is None:
            self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            print("Will run on {}".format(self.device.type))

            if param.model_weight_file != "":
                self.weights = param.model_weight_file
                label_data = param.class_file
            else:
                weights_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
                if not os.path.isdir(weights_folder):
                    os.mkdir(weights_folder)
                self.weights = os.path.join(weights_folder, param.model_name + '.pt')
                if not os.path.isfile(self.weights):
                    download_model(param.model_name, weights_folder)
                label_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", 'coco.yaml')

            self.model = DetectMultiBackend(self.weights, device=self.device, fp16=False, data=label_data)

        # Load image
        img = letterbox(src_image, param.input_size, stride=self.model.stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device.type == 'cuda' else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Set classes
        self.classes = list(self.model.names.values())
        self.set_names(self.classes)

        half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()  # to FP16

        # Inference
        with torch.no_grad():
            pred = self.model(img, augment=False, visualize=False)

        # Apply NMS
        pred = non_max_suppression(
                        pred[0][0],
                        param.conf_thres,
                        param.iou_thres,
                        classes=None,
                        max_det=1000
                    )

        # Set output results
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], src_image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xyxy=torch.stack(xyxy).cpu().numpy().reshape(1, -1)
                    boxes = xyxy[0].flatten().tolist()
                    w = float(boxes[2] - boxes[0])
                    h = float(boxes[3] - boxes[1])
                    x = float(boxes[0])
                    y = float(boxes[1])
                    self.add_object(i, int(cls), float(conf), x, y, w, h)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloV9Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_yolo_v9"
        self.info.short_description = "Object detection with YOLOv9 models"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Wang, Chien-Yao  and Liao, Hong-Yuan Mark"
        self.info.article = "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information"
        self.info.journal = "arXiv:2402.13616"
        self.info.year = 2024
        self.info.license = "GNU General Public License v3.0"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2402.13616"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolo_v9"
        self.info.original_repository = "https://github.com/WongKinYiu/yolov9"
        # Keywords used for search
        self.info.keywords = "YOLO, object, detection, real-time, Pytorch"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create algorithm object
        return InferYoloV9(self.info.name, param)
