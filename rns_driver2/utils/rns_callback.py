import subprocess

#This function executes the rns Solver in a subprocess and fetches the resulting data from it
def SolveNeutronStar(cmd, timeout=10):
    #Here we call the subprocess. Which takes in a timeout function, such that we do not stay in an infinite loop
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout = timeout)
    except subprocess.TimeoutExpired:
        print("Your C process took to long")
        return None
    #Now we have the values as a string. We have to store them correctly
    lines = result.stdout.split()
    if lines[-1] == "nan" or lines[-1] == "-nan":
        print("Some values could not get calculated, aborting this configuration")
        return None
    #rns has two types of output, once for a rotation, once without. We make those the same here
    if float(lines[-4]) < 1:
        lines = [float(line) for line in lines[-21:]]
        lines = [cmd[2]] + lines        
    elif int(float(lines[-4])) == 1:
        lines = [float(line) for line in lines[-20:-12]] + [0.0] + [0.0] + [float(line) for line in lines[-11:]]
        lines = [cmd[2]] + lines
    else:
        print("Something weird happend, that really should not")
    return lines


