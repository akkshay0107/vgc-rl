POKEMON = {
    "liquidation,lastrespects,aquajet,protect": 0,
    "superfang,feint,followme,protect": 1,
    "scaleshot,tailwind,haze,protect": 12,
    "flareblitz,knockoff,fakeout,partingshot": 3,
    "vacuumwave,earthpower,hypervoice,bloodmoon": 4,
    "makeitrain,shadowball,powergem,trick": 5,
    "expandingforce,dazzlinggleam,terablast,trickroom": 6,
    "psychic,followme,helpinghand,trickroom": 7,
    "eruption,heatwave,earthpower,weatherball": 8,
    "facade,earthquake,headlongrush,protect": 9,
    "sacredsword,psychocut,wideguard,trickroom": 10,
    "populationbomb,followme,taunt,protect": 11,
    "makeitrain,shadowball,nastyplot,protect": 13,
    "bloodmoon,earthpower,hypervoice,vacuumwave": 14,
    "drainpunch,ragefist,bulkup,protect": 15,
    "superfang,beatup,followme,protect": 16,
    "woodhammer,grassyglide,highhorsepower,fakeout": 17,
    "tailwind,moonblast,encore,faketears": 18,
    "makeitrain,shadowball,thunderbolt,powergem": 19,
    "icywind,weatherball,muddywater,hurricane": 20,
    "fakeout,grassyglide,woodhammer,taunt": 21,
    "scaleshot,lowkick,protect,tailwind": 22,
    "electroshot,dracometeor,flashcannon,protect": 23,
    "nastyplot,makeitrain,shadowball,protect": 24,
    "closecombat,direclaw,coaching,protect": 25,
    "fakeout,grassyglide,drumbeating,highhorsepower": 26,
    "scaleshot,stompingtantrum,tailwind,protect": 27,
    "blizzard,icywind,encore,auroraveil": 28,
    "rockslide,flareblitz,extremespeed,protect": 29,
    "dragonclaw,stompingtantrum,earthquake,protect": 30,
    "leafstorm,sleeppowder,encore,tailwind": 31,
    "overheat,burningjealousy,helpinghand,protect": 32,
    "scaleshot,extremespeed,icespinner,protect": 33,
    "ironhead,suckerpunch,assurance,lowkick": 34,
    "eruption,shadowball,heatwave,overheat": 35,
}



NON_VOLATILE_STATUS = {
    "none" : 0, 
    "par" : 1,
    "brn" : 2, 
    "slp" : 3, 
    "frz" : 4, 
    "psn" : 5, 
    "tox" : 6
}

TYPES = {
    "None" : 0,
    "Normal" : 1,
    "Fire" : 2,
    "Water" : 3,
    "Electric" : 4,
    "Grass" : 5,
    "Ice" : 6,
    "Fighting" : 7,
    "Poison" : 8,
    "Ground" : 9,
    "Flying" : 10,
    "Psychic" : 11,
    "Bug" : 12,
    "Rock" : 13,
    "Ghost" : 14,
    "Dragon" : 15,
    "Dark" : 16,
    "Steel" : 17,
    "Fairy" : 18,
}

def get_base_stat_multiplier(stat_stage):
    if stat_stage >= 0:
        return (2 + stat_stage) / 2
    return 2 / (2 - stat_stage)

def get_acc_ev_multiplier(stat_stage):
    if stat_stage >= 0:
        return (3 + stat_stage) / 3
    return 3 / (3 - stat_stage)
