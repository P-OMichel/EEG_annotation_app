'''
Function to determine the brain state based on the EEG signal
'''

def get_state_0_20(supp, prop_list):

    prop_delta = prop_list[0]
    prop_alpha = prop_list[1]
    prop_beta = prop_list[2]
    prop_gamma = prop_list[-1]

    prop_hf = prop_beta + prop_gamma

    #--- determine  main state

    # conditions to avoid wrong state estimation

    cond_state_3 = (prop_beta / prop_delta >= 0.5) # and prop_alpha <= 0.6

    # suppressions with either a lot of alpha supp or redundant IES
    if (supp > 0.4):
        main_state = 0
    
    # suppressions with either rare IES or redundant alpha supp
    elif (supp > 0.07):
        main_state = 1

    # if high proportion of gamma over alpha --> very Shallow or Awake
    elif prop_hf >= 0.6 or  prop_gamma >= 0.1:
        main_state = 4
    
    # if P_beta/P_alpha is higher than a threshold --> Shallow
    elif (prop_hf >= 0.15 and cond_state_3) or  prop_gamma >= 0.05 :
        main_state = 3
    
    # if none of the above criteria is met --> Ok, may not be true but if state chart correctly made then yes
    else:
        main_state = 2

    #--- determine state 

    if main_state == 0:

        if supp > 1.5:
            state = 0
        elif supp > 1:
            state = 1
        elif supp > 0.75:
            state = 2
        else:
            state = 3
    
    elif main_state == 1:

        if supp > 0.25: 
            state = 4
        elif supp > 0.15: 
            state = 5
        elif supp > 0.10: 
            state = 6
        else:
            state = 7
        
    elif main_state == 4:

        if prop_hf >= 0.8 or prop_gamma >= 0.7:
            state = 19
        elif prop_hf >= 0.70 or prop_gamma >= 0.025:
            state = 18
        else:
            state = 1

    elif main_state == 3:

        if prop_hf >= 0.5 or prop_gamma >= 0.092:
            state = 16
        elif prop_hf >= 0.40 or prop_gamma >= 0.086:
            state = 15
        elif prop_hf >= 0.31 or prop_gamma >= 0.07:
            state = 14
        elif prop_hf >= 0.22 or prop_gamma >= 0.06: 
            state = 13
        else:
            state = 12

    elif main_state == 2:


        if prop_delta >= 0.9: 
            state = 6
        elif prop_delta >= 0.75:
            if prop_alpha >= 0.3:
                state = 8
            else:
                state = 7
        elif prop_delta >= 0.6:
            state = 8         
        elif prop_delta >= 0.5:
            state = 9
        elif prop_delta >= 0.25:
            state = 10
        elif prop_delta >= 0.15:
            state = 11
        else:
            state = 12

    return state