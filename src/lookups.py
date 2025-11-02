POKEMON = {
    "liquidation,lastrespects,aquajet,protect": 1,
    "superfang,feint,followme,protect": 2,
    "scaleshot,haze,tailwind,protect": 3,
    "flareblitz,knockoff,fakeout,partingshot": 4,
    "vacuumwave,earthpower,hypervoice,bloodmoon": 5,
    "makeitrain,shadowball,powergem,trick": 6,
    "expandingforce,dazzlinggleam,terablast,trickroom": 7,
    "psychic,followme,helpinghand,trickroom": 8,
    "eruption,heatwave,earthpower,weatherball": 9,
    "facade,earthquake,headlongrush,protect": 10,
    "sacredsword,psychocut,wideguard,trickroom": 11,
    "populationbomb,followme,taunt,protect": 12,
    "scaleshot,tailwind,haze,protect": 13,
    "makeitrain,shadowball,nastyplot,protect": 14,
    "bloodmoon,earthpower,hypervoice,vacuumwave": 15,
    "drainpunch,ragefist,bulkup,protect": 16,
    "superfang,beatup,followme,protect": 17,
    "woodhammer,grassyglide,highhorsepower,fakeout": 18,
    "tailwind,moonblast,encore,faketears": 19,
    "makeitrain,shadowball,thunderbolt,powergem": 20,
    "icywind,weatherball,muddywater,hurricane": 21,
    "fakeout,grassyglide,woodhammer,taunt": 22,
    "scaleshot,lowkick,protect,tailwind": 23,
    "electroshot,dracometeor,flashcannon,protect": 24,
    "nastyplot,makeitrain,shadowball,protect": 25,
    "closecombat,direclaw,coaching,protect": 26,
    "fakeout,grassyglide,drumbeating,highhorsepower": 27,
    "scaleshot,stompingtantrum,tailwind,protect": 28,
    "blizzard,icywind,encore,auroraveil": 29,
    "rockslide,flareblitz,extremespeed,protect": 30,
    "dragonclaw,stompingtantrum,earthquake,protect": 31,
    "leafstorm,sleeppowder,encore,tailwind": 32,
    "overheat,burningjealousy,helpinghand,protect": 33,
    "scaleshot,extremespeed,icespinner,protect": 34,
    "ironhead,suckerpunch,assurance,lowkick": 35,
    "eruption,shadowball,heatwave,overheat": 36,
}

def get_base_stat_multiplier(stat_stage):
    if stat_stage >= 0:
        return (2 + stat_stage) / 2
    return 2 / (2 - stat_stage)


def get_acc_ev_multiplier(stat_stage):
    if stat_stage >= 0:
        return (3 + stat_stage) / 3
    return 3 / (3 - stat_stage)
