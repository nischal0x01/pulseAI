import os

# Create download directory
os.makedirs("../data/raw/PulseDB", exist_ok=True)
os.chdir("../data/raw/PulseDB")

# URLs for PulseDB MIMIC parts
mimic_urls = [
    # "https://rutgers.box.com/shared/static/7l8n3tn9tr0602tdss1x7e3uliahlibp.001",
    # "https://rutgers.box.com/shared/static/zco48rvz5dog72970679foen6hct15c8.002",
    # "https://rutgers.box.com/shared/static/x22qpmelx6sz3wgkm5qyc0eis429361f.003",
    # "https://rutgers.box.com/shared/static/xj25sqnluiz6s4z8tzzm5phk00ohp6e8.004",
    # "https://rutgers.box.com/shared/static/dxus2lsoop02chaspnwipwrf0g4wmenr.005",
    # "https://rutgers.box.com/shared/static/rts6sj441laenm2sy1qcemg7ke4om3j6.006",
    # "https://rutgers.box.com/shared/static/vor4hjllld7a0c3nzef8uptbb4ut3koo.007",
    # "https://rutgers.box.com/shared/static/a2qg2p4ebyrooji3z88djlokji65tlf3.008",
    # "https://rutgers.box.com/shared/static/uh6kbiuqgnib5wakiv6o35gkpusyamc7.009",
    # "https://rutgers.box.com/shared/static/h6eyhkkx48pf3ce3th1clwj43hn98j5c.010",
    # "https://rutgers.box.com/shared/static/e93dp94hxpkas45yc59n289s2wvkafgi.011",
    # "https://rutgers.box.com/shared/static/iuvyuw7dmlxvbjvt53dj49wqn3gelqni.012",
    # "https://rutgers.box.com/shared/static/qxx6tjz8c3778601ib3icu6o1rranmc7.013",
    # "https://rutgers.box.com/shared/static/ip2ninwqj8437l9fyffjprnk90ptnx9k.014",
    # "https://rutgers.box.com/shared/static/yrtbo0lg8mjhaw624iw9bbhk1obbocwd.015",
    "https://rutgers.box.com/shared/static/wmzndowgfa5xi3tvtqahxkld3ngdyjds.016"
]

# URLs for PulseDB Vital parts
vital_urls = [
    # "https://rutgers.box.com/shared/static/vtxoksmn7emeaxypb2prywgwscuefoqa.001",
    # "https://rutgers.box.com/shared/static/euzkek7c3xoy62jisheuxqar7z5y8xig.002",
    # "https://rutgers.box.com/shared/static/49lngo0benxfjw193jnqz9tctlyb3qam.003",
    # "https://rutgers.box.com/shared/static/jf4fwgkmhry20mf5tcg9t0wxvky64um0.004",
    # "https://rutgers.box.com/shared/static/2lgxysbskfuapsaan4jypvmm8316fdkc.005",
    # "https://rutgers.box.com/shared/static/x27ktb4qsx43razwo4tjmxq9v1ro0x3y.006",
    # "https://rutgers.box.com/shared/static/q0t36fikgf3pimhvnerwwnovfr0umtp8.007",
    # "https://rutgers.box.com/shared/static/ihckx2g0f981g5yz2x8v5rgwndl6yebw.008",
    # "https://rutgers.box.com/shared/static/y8j14h8tvi5b3du8nap9dnura1omfrk6.009",
    "https://rutgers.box.com/shared/static/fu0m9tx33jkxywq32shh0g8dg3not15u.010"
]

def download_files(urls, prefix):
    for i, url in enumerate(urls, 1):
        filename = f"{prefix}.zip.{str(i).zfill(3)}"
        print(f"Downloading {filename} ...")
        os.system(f"curl -L -o {filename} -C - {url}")

# Download MIMIC parts
download_files(mimic_urls, "PulseDB_MIMIC")

# Download Vital parts
download_files(vital_urls, "PulseDB_Vital")

print("All downloads started! Check files in PulseDB folder.")
