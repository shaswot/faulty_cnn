import subprocess

experiment = "mnist32-cnn_1024_256_64-1023--LIM_100"
model_instance, error_lim = experiment.split("--")

exp_script_list = [ 
                    "all-mnist32-cnn--op--ERR3 ",
                  ]

for script_name in exp_script_list:
    cmd_script = "exp_name="+script_name + " model_instance="+model_instance + " error_lim="+error_lim +" bash dispatch.sh"
    print(cmd_script)
    subprocess.call(cmd_script, shell=True)
    
