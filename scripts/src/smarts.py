#Definitions of functional groups using SMARTS strings

alkane = '[CX4;H3,H2]'
alkene = '[CX3]=[CX3]'
arene = '[cX3]1[cX3][cX3][cX3][cX3][cX3]1'
halide = '[#6][F,Cl,Br,I]'
alcohol = '[#6][OX2H]'
aldehyde = '[CX3H1](=O)[#6,H]'
ketone = '[#6][CX3](=O)[#6]'
carboxylic_acid = '[CX3](=O)[OX2H]'
acyl_halide = '[CX3](=[OX1])[F,Cl,Br,I]'
ester = '[#6][CX3](=O)[OX2H0][#6]'
ether = '[OD2]([#6])[#6]'
amine = '[NX3;$(N-[#6]);!$(N-[!#6;!#1]);!$(N-C=[O,N,S])]'
amide = '[NX3][CX3](=[OX1])[#6]'
nitrile = '[NX1]#[CX2]'
phenol = '[OX2H][cX3]:[c]'
nitro = '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'

fg_list = [
        alkane,
        alkene,
        arene, 
        halide, 
        alcohol,
        aldehyde,
        ketone, 
        carboxylic_acid, 
        acyl_halide, 
        ester, 
        ether, 
        amine, 
        amide, 
        nitrile, 
        phenol,
        nitro]

fg_names = [
        'Alkane',
        'Alkene',
        'Arene', 
        'Halide', 
        'Alcohol',
        'Aldehyde',
        'Ketone', 
        'Carboxylic acid', 
        'Acyl halide', 
        'Ester', 
        'Ether', 
        'Amine', 
        'Amide', 
        'Nitrile', 
        'Phenol',
        'Nitro']

purity_classes = [
        'Pure substance',
        'Aldehyde + Acid',
        'Aldehyde + Alcohol',
        'Molecule + Water',
        'Ether + Alcohol',
        'Amide + Ester',
        'Amide + Acid',
        'Amine + Phenol',
        'Amine + Halide',
        'Acyl halide + Acid',
        'Halide + Alcohol',
        'Halide + Amine',
        'Alcohol + Aldehyde',
        'Alcohol + Ketone',
        'Ester + Acid', 
        'Nitrile + Acid']
