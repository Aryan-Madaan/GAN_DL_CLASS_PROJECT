import numpy as np
import json
from parser_img import data_parser

def hair_desc(attr,attr_list,attr_dict):
    '''
        Caption 1
        This function will generate captions realted to beard and hair style.
        Attributes Covered :- Hair Style, Beard Mustache Style, Hair Colour.
        Some Examples are given below
    '''
    # cap_male1 = "He has a Van Dyke beard with long black hair."
    # cap_male1 = "He has a anchor beard and he is fully bald."
    # cap_male1 = "His face is shaved  and he has medium length brown hair"
    # cap_male2 = "His face is shaved and he is half bald."
    # cap_male3 = "He has a anchor beard with medium length black hair."
    # cap_female1 = "She has medium length black hair."
    # cap_female1 = "She has black hair with boy cut."

    cap = ""
    if attr[88]:
        if attr[71]:
            cap = cap + "His face is shaved "
        else:
            cap = cap + "He has a "
            for i in range(70,86):
                if attr[i]:
                    cap = cap + attr_dict[attr_list[i]]
            # if not attr[74] or not attr[75] or not attr[76] or not attr[80] or not attr[83]:
            cap = cap + " "
        if attr[66] or attr[69]:
            cap = cap + "and he is "
            if attr[66]:
                cap = cap + "half bald."
            if attr[69]:
                cap = cap + "fully bald."
        else:
            cap = cap + "and he has "
            if not attr[68]:
                for i in range(64,69):
                    if attr[i]:
                        cap = cap + attr_dict[attr_list[i]]
                cap = cap + " "
                for i in range(53,58):
                    if attr[i]:
                        cap = cap + attr_dict[attr_list[i]]
                cap = cap + " hair."
            else:
                cap = cap + "a "
                for i in range(53,58):
                    if attr[i]:
                        cap = cap + attr_dict[attr_list[i]]
                cap = cap + " "
                for i in range(64,69):
                    if attr[i]:
                        cap = cap + attr_dict[attr_list[i]]
                cap = cap + "."

    else:
        if attr[66] or attr[69]:
            cap = "He is "
            if attr[66]:
                cap = cap + "half bald."
            if attr[69]:
                cap = cap + "fully bald."
        else:
            cap = cap + "He has "
            if not attr[68]:
                for i in range(64,69):
                    if attr[i]:
                        cap = cap + attr_dict[attr_list[i]]
                cap = cap + " "
                for i in range(53,58):
                    if attr[i]:
                        cap = cap + attr_dict[attr_list[i]]
                cap = cap + " hair."
            else:
                cap = cap + "a "
                for i in range(53,58):
                    if attr[i]:
                        cap = cap + attr_dict[attr_list[i]]
                cap = cap + " "
                for i in range(64,69):
                    if attr[i]:
                        cap = cap + attr_dict[attr_list[i]]
                cap = cap + "."
    return cap    

def lip_forehead_desc(attr,attr_list,attr_dict):
    '''
        Caption 2
        This function will generate captions realted to lip and forehead.
        Attributes Covered :- Type of Lip, Color of Lip, Forehead. 
        Some Examples are given below
    '''
    # cap1 = "He has thin pink lips and a curved forehead."
    # cap1 = "He has full red lips and a fuzzi mount forehead."

    cap = "He has "
    for i in range(7,13):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]
    cap = cap +" "
    for i in range(13,17):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]
    cap = cap + " coloured lips and a "
    
    for i in range(17,23):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]
    
    cap = cap + " forehead."
    return cap


def gender_age_desc(attr,attr_list,attr_dict):
    '''
        Caption 3
        This function will generate captions realted to age, gender, face shape and jaw ling.
        Attributes Covered :- Face Shape, Age (Rough estimate as appearing), Gender, Jaw Line. 
        For all with age <30 yrs are young, above 30 and below 45 we have classified them as middle aged.
        Some Examples are given below
    '''
    # cap1 = "A young male with a round face and flat jaw line."
    # cap1 = "A middle aged female with a oblong face and sharp jaw line." 

    cap = ""

    if attr[87]:
        cap = cap + "A young "
    else:
        cap = cap + "A middle aged "
    
    if attr[88]:
        cap = cap + "male "
    else:
        cap = cap + "female "
    
    cap = cap + "with a "

    for i in range(1,7):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]

    cap = cap + " shaped face and "

    for i in range(46,49):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]
    
    cap = cap + " jaw line."

    return cap


def ear_skin_desc(attr,attr_list,attr_dict):
    '''
        Caption 4
        This function will generate captions realted to ear, skin colour, nose and dimples .
        Attributes Covered :- Cheeks with Dimple, Ear, Skin Color, Nose. 
        Some Examples are given below
    '''
    # cap1 = "He has a straigt nose, pointed ear and his skin colour is fair."
    # cap1 = "He has a straigt nose, broad lobed ear, his skin colour is brown and he has dimples on his cheek."
    cap = "He has a "

    for i in range(58,64):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]

    cap = cap + " nose, "

    for i in range(24,28):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]
    
    if attr[23]:
        cap = cap + " ear, his skin colour is "
        for i in range(49,53):
            if attr[i]:
                cap = cap + attr_dict[attr_list[i]]
        cap = cap + " and he has dimples on his cheeks."
    else:
        cap = cap + " ear and his skin colour is "
        for i in range(49,53):
            if attr[i]:
                cap = cap + attr_dict[attr_list[i]]
    
    cap = cap+"."
    return cap

def eyes_desc(attr,attr_list,attr_dict):
    '''
        Caption 5
        This function will generate captions realted to eyes.
        Attributes Covered :- Eye Size, Eye Defect, Eye Color. 
        Some Examples are given below
    '''
    # cap1 = "He has medium sized black eyes with no visible defect."
    # cap1 = "He has large sized black eyes with a defect in right eye."
    cap = "He has "

    for i in range(28,31):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]

    cap = cap + " sized "

    for i in range(35,40):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]
    
    cap = cap + " coloured eyes with "

    for i in range(31,35):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]
    
    cap = cap + " ."

    return cap
    

def eyebrow_desc(attr,attr_list,attr_dict):
    '''
        Caption 5
        This function will generate captions realted to eyebrows.
        Attributes Covered :- Eyebrow Style, Eyebrow Separation. 
        Some Examples are given below
    '''
    # cap1 = "He has very thick and highly separated eyebrows."
    cap = "He has "

    for i in range(40,43):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]
    
    cap = cap + " and "

    for i in range(43,46):
        if attr[i]:
            cap = cap + attr_dict[attr_list[i]]

    cap = cap + " eyebrows."

    return cap
    


def clean(s):
    if len(s) == 0:
        return s

    s = s.replace('.', '').strip()
    a = ''

    for i in [x + ' ' for x in s.split(' ') if len(x)>0]:
        a += i

    a = a.strip()
    a += '.'

    return a

def main(attr, attr_list):
    attr_dict = json.load(open('dict.txt','r'))
    captions = ['','','','','','','','']
    fname = attr[0]
    attr = data_parser(attr)
    ind = 1
    # print(len(attr))
    # print(attr_dict)

    captions[1] = gender_age_desc(attr,attr_list,attr_dict)
    captions[2] = lip_forehead_desc(attr,attr_list,attr_dict)
    captions[3] = ear_skin_desc(attr,attr_list,attr_dict)
    captions[4] = eyes_desc(attr,attr_list,attr_dict)
    captions[5] = eyebrow_desc(attr,attr_list,attr_dict)
    captions[6] = hair_desc(attr,attr_list,attr_dict)

    '''
        Changing the gender according to the given data, Currently takes 2 genders into considerations. (Male or Female)
    '''

    if not attr[88]:
        captions = [captions[i].replace('man','woman') for i in range(0,len(captions))]
        captions = [captions[i].replace('He','She') for i in range(0,len(captions))]
        captions = [captions[i].replace('His','Her') for i in range(0,len(captions))]
        captions = [captions[i].replace('his','her') for i in range(0,len(captions))]

    '''
        Writing Data to file, format is Image File Name and then Caption.
        It writes to file:- caps.txt.
        Example Output:- 
        S001.jpeg "A young male with a rectangular shaped face and flat jaw line."
    '''
    f.write(fname + '.jpeg "')
    last = -1
    #
    for i, c in enumerate(captions):
        if len(c) > 0:
            last = max(last, i)

    for i, c in enumerate(captions):
        if len(c) > 0:
            f.write(clean(c))
            if i != last:
                f.write(' ')

    f.write('"\n')

file = open('list_attr_celeba.txt') # Opening the one hot encoded and cleaned txt file.
f = open('caps.txt', 'w') # File where the captions will be saved.
attr_list =[]
attr = []
for line in file:
    '''
        Getting the attributes of the first image and storing them in a attr_list
    '''
    attr_list = line.split(' ')
    # attr_list = attr_list[:-1]
    break
print(attr_list)  
for line in file:
    attr = line.split(' ')
    # print(attr)
    main(attr, attr_list)
    # break