import sys
#print("here is tool/train.py")
#sys.path.append('/gemini/code/Motion-2Dto3D')
print(sys.path)
import hydra
from hmr4d.train import train
#print(train)

@hydra.main(version_base=None, config_path="../hmr4d/configs", config_name="train")
def main(cfg) -> None:
    #print("training started ...")
    train(cfg)
    
if __name__ == "__main__":
    main()