
module Chemistry

export atomic_number,
       chemical_symbol,
       atomic_mass,
       element_name,
       rnn

# TODO: what else should be in here?
#    - crystal structure(s)
#    - unit cell
#    -

# TODO: Weight should be atomic mass - probably these are incorrect units
# Z	Symbol	Name	Weight
elements_table = [
1	   :H	   "Hydrogen"	     1.00794
2	   :He	"Helium"	      4.002602
3	   :Li	"Lithium"	     6.941
4	   :Be	"Beryllium"	      9.012182
5	   :B	   "Boron"	     10.811
6	   :C	   "Carbon"	    12.0107
7	   :N	   "Nitrogen"	     14.0067
8	   :O	   "Oxygen"	    15.9994
9	   :F	   "Fluorine"	     18.9984032
10	   :Ne	"Neon"	    20.1797
11	   :Na	"Sodium"	     22.989770
12	   :Mg	"Magnesium"	     24.3050
13	   :Al	"Aluminium"	     26.981538
14	   :Si	"Silicon"	    28.0855
15	   :P	   "Phosphorus"	     30.973761
16	   :S	   "Sulfur"	      32.065
17	   :Cl	"Chlorine"	      35.453
18	   :Ar	"Argon"	      39.948
19	   :K	   "Potassium"	      39.0983
20	   :Ca	"Calcium"	    40.078
21	   :Sc	"Scandium"	      44.955910
22	   :Ti	"Titanium"	      47.867
23	   :V	   "Vanadium"	    50.9415
24	   :Cr	"Chromium"	      51.9961
25	   :Mn	"Manganese"	     54.938049
26	   :Fe	"Iron"	    55.845
27	   :Co	"Cobalt"	     58.933200
28	   :Ni	"Nickel"	     58.6934
29	   :Cu	"Copper"	     63.546
30	   :Zn	"Zinc"	    65.38
31	   :Ga	"Gallium"	    69.723
32	   :Ge	"Germanium"	     72.64
33	   :As	"Arsenic"	    74.92160
34	   :Se	"Selenium"	      78.96
35	   :Br	"Bromine"	    79.904
36	   :Kr	"Krypton"	    83.798
37	   :Rb	"Rubidium"	      85.4678
38	   :Sr	"Strontium"	     87.62
39	   :Y	   "Yttrium"	     88.90585
40	   :Zr	"Zirconium"	     91.224
41	   :Nb	"Niobium"	    92.90638
42	   :Mo	"Molybdenum"	    95.94
43	   :Tc	"Technetium"	    98
44	   :Ru	"Ruthenium"	     101.07
45	   :Rh	"Rhodium"	    102.90550
46	   :Pd	"Palladium"	     106.42
47	   :Ag	"Silver"	     107.8682
48	   :Cd	"Cadmium"	    112.411
49	   :In	"Indium"	     114.818
50	   :Sn	"Tin"	     118.710
51	   :Sb	"Antimony"	      121.760
52	   :Te	"Tellurium"	     127.60
53	   :I	   "Iodine"	      126.90447
54	   :Xe	"Xenon"	      131.293
55	   :Cs	"Cesium"	     132.90545
56	   :Ba	"Barium"	     137.327
57	   :La	"Lanthanum"	     138.9055
58	   :Ce	"Cerium"	     140.116
59	   :Pr	"Praseodymium"	     140.90765
60	   :Nd	"Neodymium"	     144.24
61	   :Pm	"Promethium"	    145
62	   :Sm	"Samarium"	      150.36
63	   :Eu	"Europium"	      151.964
64	   :Gd	"Gadolinium"	    157.25
65	   :Tb	"Terbium"	    158.92534
66	   :Dy	"Dysprosium"	    162.500
67	   :Ho	"Holmium"	    164.93032
68	   :Er	"Erbium"	     167.259
69	   :Tm	"Thulium"	    168.93421
70	   :Yb	"Ytterbium"	     173.04
71	   :Lu	"Lutetium"	      174.967
72	   :Hf	"Hafnium"	    178.49
73	   :Ta	"Tantalum"	      180.9479
74	   :W	   "Tungsten"	    183.84
75	   :Re	"Rhenium"	    186.207
76	   :Os	"Osmium"	     190.23
77	   :Ir	"Iridium"	    192.217
78	   :Pt	"Platinum"	      195.078
79	   :Au	"Gold"	    196.96655
80	   :Hg	"Mercury"	    200.59
81	   :Tl	"Thallium"	      204.3833
82	   :Pb	"Lead"	    207.2
83	   :Bi	"Bismuth"	    208.98038
84	   :Po	"Polonium"	      209
85	   :At	"Astatine"	      210
86	   :Rn	"Radon"	      222
87	   :Fr	"Francium"	      223
88	   :Ra	"Radium"	     226
89	   :Ac	"Actinium"	      227
90	   :Th	"Thorium"	    232.0381
91	   :Pa	"Protactinium"	     231.03588
92	   :U	   "Uranium"	     238.02891
93	   :Np	"Neptunium"	     237
94	   :Pu	"Plutonium"	     244
95	   :Am	"Americium"	     243
96	   :Cm	"Curium"	     247
97	   :Bk	"Berkelium"	     247
98	   :Cf	"Californium"	      251
99	   :Es	"Einsteinium"	      252
100	:Fm	"Fermium"	      257
101	:Md	"Mendelevium"	     258
102	:No	"Nobelium"	     259
103	:Lr	"Lawrencium"	      262
104	:Rf	"Rutherfordium"	      261
105	:Db	"Dubnium"	      262
106	:Sg	"Seaborgium"	      266
107	:Hs	"Hassium"	      264
108	:Bh	"Bohrium"	      277
109	:Mt	"Meitnerium"	      268
110	:Uun	"Ununnilium"	     281
111	:Uuu	"Unununium"	      272
112	:Uub	"Ununbium"	    285
113	:Uut	"Ununtrium"	      284
114	:Uuq	"Ununquadium"	    289
115	:Uup	"Ununpentium"	    288
116	:Uuh	"Ununhexium"	     292 ]

# TODO: the following data is just imported from ASE, this is quite the hack,
# it would be much better to have a script that automatically imports all
# interesting data, including crystal structures, etc from ASE and
# make it available in a databse, either file, or Julia code.
# _rnn = Float64[]
# for n = 1:116
#    r0 = -1.0
#    try
#       r0 = rnn(JuLIP.Chemistry.chemical_symbol(n))
#    catch
#    end
#    push!(_rnn, r0)
# end
# # the rnn function used here is
# """
# `rnn(species)` : returns the nearest-neighbour distance for a given species
# """
# function rnn(species::Symbol)
#    X = positions(bulk(species) * 2)
#    return minimum( norm(X[n]-X[m]) for n = 1:length(X) for m = n+1:length(X) )
# end
#
# rnn(s::AbstractString) = rnn(Symbol(s))
#
const _rnn = [-1.0, -1.0, 3.02243, 2.22873, -1.0, 1.54586, -1.0, -1.0, -1.0,
   3.13248, 3.66329, 3.19823, 2.86378, 2.35126, -1.0, -1.0, -1.0, 3.71938,
   4.52931, 3.94566, 3.25752, 2.89607, 2.6154, 2.49415, -1.0, 2.48549, 2.49875,
   2.48902, 2.55266, 2.66, -1.0, 2.45085, -1.0, 3.53122, -1.0, 4.04465, 4.84108,
   4.29921, 3.55822, 3.17748, 2.85788, 2.72798, 2.70767, 2.64627, 2.68701,
   2.75065, 2.89207, 2.98, -1.0, -1.0, -1.0, 3.91893, -1.0, 4.38406, 5.23945,
   4.34745, 3.72861, 3.64867, 3.6416, 3.63168, -1.0, -1.0, 3.99238, 3.57345,
   3.524, 3.50263, 3.48854, 3.46905, 3.44956, 3.88202, 3.44157, 3.13374,
   2.86654, 2.73664, 2.73976, 2.67994, 2.71529, 2.77186, 2.885, -1.0, 3.41215,
   3.50018, -1.0, 3.35, -1.0, -1.0, -1.0, -1.0, 3.75474, 3.5921, -1.0, -1.0,
   -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
   -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,-1.0, -1.0, -1.0, -1.0]

const _symbols = Vector{Symbol}(elements_table[:, 2])
const _names = Vector{String}(elements_table[:, 3])
const _masses = Vector{Float64}(elements_table[:, 4])
const _numbers = Dict{Symbol, Int}()

for (n, sym) in enumerate(_symbols)
   _numbers[sym] = n
end


atomic_number(sym::Symbol) = _numbers[sym]

chemical_symbol(z::Integer) = _symbols[z]

atomic_mass(z::Integer) = _masses[n]
atomic_mass(sym::Symbol) = atomic_mass(atomic_number(sym))

element_name(z::Integer) = _names[n]
element_name(sym::Symbol) = element_name(atomic_number(sym))

rnn(z::Integer) = _rnn[z]
rnn(sym::Symbol) = rnn(atomic_number(sym))


end
