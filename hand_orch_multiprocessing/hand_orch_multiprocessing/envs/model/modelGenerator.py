#!/usr/bin/env python
import yaml
import random
import io
import subprocess
from pkg_resources import resource_string,resource_filename


import re
class DomainRandomization():
    def __init__(self,path=None,load_ws=False,load_ws_pcd = False):
        self.load_ws = load_ws
        self.load_ws_pcd = load_ws_pcd 
        if path == None:
            self.config_path = "./config.yml"
            self.shell_path = "./modelGenerator.sh"
            self.lib_path =  ""
            self.folder_path =""
        else:
            self.config_path = path+"/model/config.yml"
            self.shell_path = resource_filename(__name__,"/modelGenerator.sh")
            self.lib_path =  resource_filename(__name__,"")
            self.folder_path = path+"/model"

        if self.folder_path =="":
            self.folder_path ="."
        if self.lib_path =="":
            self.lib_path =""
        print("self.folder_path::",self.folder_path)
        print("self.lib_path::",self.lib_path)

        with open(self.config_path, 'r') as stream:
            try:
                self.config_dic = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc) 
    

    def visual_randomization(self):
        """
        randomize the color values:
            ambient
            diffuse
            specular
        """
        def color_randomizer():
            color = [0,0,0,1]
            for i in range(len(color)-1):
                color[i]  = random.uniform(0, 1)
            return color
    
        config_dic_copy = self.config_dic
        for hand_part,_ in self.config_dic.items():
            links = self.config_dic[hand_part]["Links"]
            for link,_ in links.items():
                #print(link)
                #print(config_dic_copy[hand_part]["Links"][link]["color"]["ambient"])
                if link !="thhub":
                    config_dic_copy[hand_part]["Links"][link]["color"]["ambient"]  = color_randomizer()
                    config_dic_copy[hand_part]["Links"][link]["color"]["diffuse"]  = color_randomizer()
                    config_dic_copy[hand_part]["Links"][link]["color"]["specular"] = color_randomizer()
        self.config_dic = config_dic_copy

    def physic_prop_randomizer(self,damping_limit =[0.1-0.01,0.1+0.1],friction_limit=[0,0+0.01]):
        """
        changes damping and friction 
            input: 
                -damping_limit  = [lower_limit,upper_limit]
                -friction_limit = [lower_limit,upper_limit]
        """
        def damping_randomizer(damping_limit):
            return random.uniform(damping_limit[0],damping_limit[1])
        def friction_randomizer(friction_limit):
            return random.uniform(friction_limit[0],friction_limit[1])
        
        config_dic_copy = self.config_dic
        for hand_part,_ in self.config_dic.iteritems():
            links = self.config_dic[hand_part]["Links"]
            for link,_ in links.iteritems():
                #print(link)
                #print(config_dic_copy[hand_part]["Links"][link]["color"]["ambient"])
                if link !="forearm":
                    config_dic_copy[hand_part]["Links"][link]["joint"]["damping"]  = damping_randomizer(damping_limit)
                    config_dic_copy[hand_part]["Links"][link]["joint"]["friction"] = friction_randomizer(friction_limit)
       
        self.config_dic = config_dic_copy
   
    def get_config():
        return self.config_dic

    def save_setting(self,config_dic = None):
        if config_dic !=None:
            self.config_dic = config_dic
    
        
        #dump dic to file
        with io.open(self.config_path, 'w', encoding='utf8') as outfile:
            yaml.dump(self.config_dic, outfile)
    
    def launch_model_sdf(self):
        print("calling::launch_model_sdf")
        subprocess.call(['bash', self.shell_path, 'false', 'true',str(self.folder_path),str(self.lib_path),self.str_bool(self.load_ws),self.str_bool(self.load_ws_pcd)])
    
    def generate_model_sdf(self,control_mode="full"):
        print("\n\n")
        print("self.load_ws::",self.load_ws)
        print("calling::generate_model_sdf")
        print("\n\n")
        subprocess.call(['bash', self.shell_path, 'true', 'false',str(self.folder_path),str(self.lib_path),self.str_bool(self.load_ws),self.str_bool(self.load_ws_pcd),control_mode])
    
    def generateAndLaunch_model_sdf(self):
        print("calling::generateAndLaunch_model_sdf")
        subprocess.call(['bash', self.shell_path, 'true', 'true' ,str(self.folder_path),str(self.lib_path),self.str_bool(self.load_ws),self.str_bool(self.load_ws_pcd)])

    def str_bool(self,boolean):
        if boolean:
            return "true"
        else:
            return "false"
            


# dr = DomainRandomization()
# dr.visual_randomization()
#dr.physic_prop_randomizer()
# dr.save_setting()

#dr.launch_model_sdf()
# dr.generate_model_sdf()
# dr.generateAndLaunch_model_sdf()



