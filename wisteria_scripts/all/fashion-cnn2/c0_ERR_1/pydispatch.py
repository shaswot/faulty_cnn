import subprocess

experiment = "fashion-cnn2_1024-1023--LIM_500"
model_instance, error_lim = experiment.split("--")

exp_script_list = [ 
                    "all-fashion-cnn2--c0--ERR_1",
                  ]

for script_name in exp_script_list:
    cmd_script = "exp_name="+script_name + " model_instance="+model_instance + " error_lim="+error_lim +" bash dispatch.sh"
    print(cmd_script)
    subprocess.call(cmd_script, shell=True)
    
