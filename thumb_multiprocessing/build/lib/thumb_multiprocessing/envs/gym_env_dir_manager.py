import os
import shutil 
from pkg_resources import resource_string,resource_filename
class ENV_dir_manager():
  def __init__(self):
    self.path = None

  
  def setup(self,path):
    
    res =self.set_path(path)
    if res:
      self.copy_initail_gym_setting_to_path(path)
      self.path = path
    else:
      print("could not copy initial setting of env")
  def creat_path_if_does_not_exist(self,path):
    if os.path.isdir(path):
      res = True
    else:
      res = self.create_path(path)

    return res

  def create_path(self,path):
    try:
      res = os.makedirs(path) 
      return True
    except:
      return False

  def set_path(self,path):
    """
    changes the path of env variable
    """
    os.environ["KUKA_HANDLIT_DIR"] = path
    res = self.creat_path_if_does_not_exist(path)
    if(res):
      print("path succesfully created at: ",path)
      return path
    else:
      print("path creation has failed")
      return False

  def copy_initail_gym_setting_to_path(self,path):
    if os.path.exists(path):
      shutil.rmtree(path) 
    dist = path
    src =  resource_filename(__name__,"/initial_env_setting")
    if src == "initial_env_setting":
      src = "./initial_env_setting"
    print("src:: ",src)
    self.copytree(src, dist)

  def copytree(self,src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


  def prompt(self):
    self.path = input("could you provide a path for env_setting:(This is normaly the same path as where u store your training)")
    self.setup(self.path)



# env_dir_manager =ENV_dir_manager()

