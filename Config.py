import json
import configparser
from pyexpat import model


class Config():
    def __init__(self):
        self.defaultjson = '/home/vision/diska4/shy/SAM/data/dataset_0.json'
        # self.config = configparser().ConfigParser()

    def parse_json(self, json_dir=None):
        if json_dir == None:
            file_pth = self.defaultjson
        else:
            file_pth = json_dir

        with open(file_pth, 'r') as file:
            self.json_data = json.load(file)

    def parse_Data_json(self, json_dir=None):
        self.parse_json(json_dir)

        self.description = self.json_data['description']
        self.labels = self.json_data['labels']
        self.modality = self.json_data['modality']
        self.name = self.json_data['name']
        self.numTest = int(self.json_data['numTest'])
        self.numTraining = int(self.json_data['numTraining'])
        self.tensorImageSize = self.json_data['tensorImageSize']

        self.test = self.json_data['test']
        self.training = self.json_data['training']
        self.validation = self.json_data['validation']

        self.test = [inp.split('/')[1] for inp in self.test]
        self.training = [{"image":d['image'].split('/')[1], "label":d['label'].split('/')[1]} for d in self.training]
        self.validation = [{"image":d['image'].split('/')[1], "label":d['label'].split('/')[1]} for d in self.validation]

        return self
    def parse_Basic_json(self, json_dir=None):
        self.parse_json(json_dir)

        self.device = self.json_data['device']
        self.model_type = self.json_data['model_type']
        self.seg_type = self.json_data['seg_type']
        self.sam_checkpoint = self.json_data['sam_checkpoint']
        self.train_path = self.json_data['train_path']
        self.test_path = self.json_data['test_path']
        self.valid_path = self.json_data['valid_path']
        return self


class Log:
    def __init__(self, file:str) -> None:
        self.f = open(file=file, mode='w+')
        
    def format_flush(args:dict):
        pass




CONFIG = Config()
# config.parse_Data_json()

if __name__ == '__main__':
    print(CONFIG.training)