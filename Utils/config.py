import os
import json

class Config:
    def __init__(self, file_name, temp_directory):
        self.temp_directory = temp_directory
        self.config = {}
        data = None
        with open(file_name) as json_file:
            data = json.load(json_file)

        assert(data is not None)
        self.config['inference_engines'] = []
        for inference_engine in data['inference_engines']:
            self.create_inference_config_file(inference_engine)
        self.config['video_source'] = data['video_source']
        self.config['sink_type'] = data['sink_type']
        self.config['tracker'] = data['tracker']
        self.config['va_filters'] = data['va_filters']

    def get_config(self):
        return self.config

    def create_inference_config_file(self, inference_engine_config):
        file_name = os.path.join(self.temp_directory, inference_engine_config['name'] + 'config.txt')
        property = 'property'
        class_atr = 'class-attrs-all'

        if property in inference_engine_config:
            with open(file_name, 'w') as f:
                f.write('[' + property + '] \n')
                for property_name in inference_engine_config[property]:
                    f.write(property_name + "=" + str(inference_engine_config[property][property_name]) + '\n')
                f.write('\n')
                if class_atr in inference_engine_config:
                    f.write('[' + class_atr + '] \n')
                    for class_atr_name in inference_engine_config[class_atr]:

                        f.write(class_atr_name + "=" + str(inference_engine_config[class_atr][class_atr_name]) + '\n')
            #self.inference_config_file_list.append((file_name))
            self.config['inference_engines'].append(
                {
                    "name": inference_engine_config['name'],
                    "spec_file": file_name
                })
            print(inference_engine_config['name'])

        else:
            print(file_name + " could not be ceated due to missing data in configuration file")