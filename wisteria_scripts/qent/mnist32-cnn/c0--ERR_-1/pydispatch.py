import subprocess

experiment = "mnist32-cnn_1024_256_64-1023--LIM_05"
model_instance, error_lim = experiment.split("--")

exp_script_list = [ 
                    "qent1-mnist32-cnn--c0--ERR_-1",
                    "qent2-mnist32-cnn--c0--ERR_-1",
                    "qent3-mnist32-cnn--c0--ERR_-1",
                    "qent4-mnist32-cnn--c0--ERR_-1"
                  ]

for script_name in exp_script_list:
    cmd_script = "exp_name="+script_name + " model_instance="+model_instance + " error_lim="+error_lim +" bash dispatch.sh"
    print(cmd_script)
    subprocess.call(cmd_script, shell=True)
    
