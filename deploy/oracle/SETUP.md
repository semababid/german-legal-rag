# Putting this on a free Oracle VM

The result is a link like `https://mietrecht-rag.duckdns.org` with a real
certificate, running on hardware that stays free for the life of the account.
Budget about two hours the first time, most of it waiting.

Oracle Always Free gives 2 Arm cores and 12 GB of memory, which is four times
what this needs. The catch is that it is a plain virtual machine: nothing is
managed for you, and two of the steps below exist only because Oracle behaves
differently from every other host.

## 1. Create the account

[cloud.oracle.com/free](https://cloud.oracle.com/free). A credit card is
required for identity verification. Always Free resources are not charged, but
you do have to enter one.

Pick your home region carefully. It cannot be changed later, and Arm capacity
varies by region. Frankfurt and Amsterdam are close to Germany; if one has no
capacity, the other may.

## 2. Create the instance

Compute → Instances → Create instance.

- **Image**: Canonical Ubuntu 24.04
- **Shape**: change it. The default is an AMD micro with 1 GB, which is not
  enough. Choose **Ampere** → `VM.Standard.A1.Flex`, then **2 OCPUs and 12 GB**.
- **SSH keys**: upload your public key, or let it generate one and download it.
  Without this you cannot get in.

If you see **"Out of host capacity"**, that is Oracle, not you. Arm capacity in
popular regions runs out regularly. Try a different availability domain, or try
again later; it often frees up within a day.

## 3. Open the ports, in both places

This is where most Oracle deployments stall. There are two firewalls and both
block by default, so opening one and testing gets you a timeout that looks like
a broken app.

**The cloud firewall.** Networking → Virtual Cloud Networks → your VCN →
Security Lists → Default. Add two ingress rules:

| Source | Protocol | Destination port |
| --- | --- | --- |
| 0.0.0.0/0 | TCP | 80 |
| 0.0.0.0/0 | TCP | 443 |

**The firewall inside the machine.** Oracle's Ubuntu images ship iptables rules
that reject almost everything, which no other cloud does. Over ssh:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## 4. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

Log out and back in, so your user can talk to Docker without sudo.

## 5. Point a name at the machine

[duckdns.org](https://www.duckdns.org): sign in with GitHub or Google, pick a
subdomain, and set its IP to the instance's public address. Free, and it takes a
minute.

Check it resolves before going further, because Let's Encrypt will not issue a
certificate for a name that does not point here yet:

```bash
dig +short mietrecht-rag.duckdns.org
```

## 6. Set your hostname and upload

On your laptop, edit `deploy/oracle/Caddyfile` and replace the example hostname
with yours. Then:

```bash
./deploy/oracle/upload.sh ubuntu@<your-instance-ip>
```

That copies the source, the page, the corpus and the 49 MB index, builds the
image on the VM and starts it. The first build installs torch and takes a while;
the first start downloads bge-m3, about 2.3 GB, into a volume that survives
restarts.

## 7. Watch it come up

```bash
ssh ubuntu@<ip> 'cd ~/german-legal-rag && docker compose logs -f'
```

You want two things in the log: `"event": "startup"` from the API, and a line
from Caddy about obtaining a certificate. Then open your link.

## Keeping it alive

Oracle reclaims instances that look abandoned: over a 7 day window, CPU below
20%, network below 20% **and** memory below 20%. A demo nobody visits is exactly
that shape, except for memory. bge-m3 holds roughly 3 GB resident, which against
12 GB is 25%, so the memory test alone should keep it. If you want more margin,
recreate the instance with 6 GB instead of 12 and the same model becomes half of
allocated memory.

## When something is wrong

**The link times out.** Almost always step 3, and almost always the iptables
half rather than the cloud half. Check from the VM that the app answers locally
first: `curl -sI localhost:80`. If that works and the outside does not, it is
the firewall.

**No certificate, Caddy retrying.** The name has to resolve to this machine
*before* Caddy asks. Fix DNS, then `docker compose restart caddy`.

**The API says unhealthy for several minutes after a restart.** Expected. It is
loading the embedding model, which takes about thirty seconds on this hardware,
and the health check allows three minutes before it starts complaining.

**Out of memory during the build.** Building torch wheels is the heaviest moment.
If it dies, add swap: `sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
&& sudo mkswap /swapfile && sudo swapon /swapfile`.

## What this deployment does and does not do

It runs retrieval, not generation. Twelve gigabytes would hold the 9B model, but
two Arm cores would take minutes per answer, and a model small enough to be
quick would answer worse than the retrieval it sits on. The page asks `/health`
on load and stops offering answers it cannot write.

Questions are kept out of the log here, unlike the local setup. A public demo
should not accumulate strangers' tenancy problems on a disk you own.
