import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
# pyrefly: ignore [missing-import]
from groq import Groq
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

# ── In-memory cache ──────────────────────────────────────────
_cache: dict = {}

def _cache_key(attack_type: str, details: dict) -> str:
    raw = f"{attack_type}:{json.dumps(details, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()

# ── Prompt Templates ─────────────────────────────────────────
PROMPT_TEMPLATES = {
    'DDoS': """You are a senior cybersecurity analyst.
A DDoS attack has been detected with these metrics:
- Flows per second: {flows_per_sec}
- Packet rate: {packet_rate}
- Source IPs: {source_ips}
- Duration: {duration} seconds

Write a concise threat brief (max 150 words) covering:
1. Attack severity (Low/Medium/High/Critical)
2. What the attacker is trying to do
3. Immediate containment steps
4. Long-term mitigation""",

    'PortScan': """You are a senior cybersecurity analyst.
A Port Scan attack has been detected:
- Ports scanned: {ports_scanned}
- Scan rate: {scan_rate} ports/sec
- Source IP: {source_ip}
- Duration: {duration} seconds

Write a concise threat brief (max 150 words) covering:
1. Attack severity (Low/Medium/High/Critical)
2. What the attacker is trying to do
3. Immediate containment steps
4. Long-term mitigation""",

    'default': """You are a senior cybersecurity analyst.
Attack detected: {attack_type}
Metrics: {details}

Write a concise threat brief (max 150 words) covering:
1. Attack severity (Low/Medium/High/Critical)
2. What the attacker is trying to do
3. Immediate containment steps
4. Long-term mitigation"""
}

# ── Rule-based fallback (no API needed) ──────────────────────
RULE_BASED_RESPONSES = {
    'DDoS': {
        'severity': 'Critical',
        'description': 'Distributed Denial of Service attack flooding network resources.',
        'containment': 'Enable rate limiting, block source IPs, activate CDN scrubbing.',
        'mitigation': 'Deploy anti-DDoS appliance, configure BGP blackholing.'
    },
    'PortScan': {
        'severity': 'Medium',
        'description': 'Reconnaissance scan probing for open ports and vulnerabilities.',
        'containment': 'Block source IP at firewall, enable port scan detection.',
        'mitigation': 'Implement network segmentation, close unused ports.'
    },
    'DoS Hulk': {
        'severity': 'High',
        'description': 'HTTP flood attack overwhelming web server with requests.',
        'containment': 'Enable WAF rules, rate-limit HTTP requests, block attacker IP.',
        'mitigation': 'Deploy load balancer, implement CAPTCHA challenges.'
    },
    'Bot': {
        'severity': 'High',
        'description': 'Botnet activity detected — compromised host communicating with C2.',
        'containment': 'Isolate affected host, block C2 domains at DNS level.',
        'mitigation': 'Patch vulnerabilities, deploy EDR solution.'
    },
    'FTP-Patator': {
        'severity': 'Medium',
        'description': 'Brute force attack against FTP service.',
        'containment': 'Block source IP, lock affected accounts, disable FTP if unused.',
        'mitigation': 'Replace FTP with SFTP, implement account lockout policy.'
    },
    'SSH-Patator': {
        'severity': 'Medium',
        'description': 'Brute force attack against SSH service.',
        'containment': 'Block source IP, enforce key-based auth, disable password login.',
        'mitigation': 'Deploy fail2ban, change SSH port, use MFA.'
    },
    'Heartbleed': {
        'severity': 'Critical',
        'description': 'Heartbleed exploit attempt targeting OpenSSL vulnerability.',
        'containment': 'Immediately patch OpenSSL, revoke and reissue SSL certificates.',
        'mitigation': 'Upgrade to OpenSSL 1.0.1g+, audit all SSL endpoints.'
    },
    'default': {
        'severity': 'Medium',
        'description': 'Suspicious network activity detected requiring investigation.',
        'containment': 'Monitor traffic, isolate affected systems if needed.',
        'mitigation': 'Review firewall rules, update IDS signatures.'
    }
}

def get_rule_based_brief(attack_type: str) -> dict:
    response = RULE_BASED_RESPONSES.get(
        attack_type,
        RULE_BASED_RESPONSES['default']
    )
    return {
        'attack_type': attack_type,
        'timestamp':   datetime.now().isoformat(),
        'source':      'rule_based',
        'severity':    response['severity'],
        'brief': (
            f"[{response['severity']}] {response['description']} "
            f"Containment: {response['containment']} "
            f"Mitigation: {response['mitigation']}"
        )
    }

def get_groq_brief(attack_type: str, details: dict) -> dict:
    api_key = os.getenv('GROQ_API_KEY')

    # Fallback if no API key
    if not api_key:
        logger.warning("No Groq API key — using rule-based fallback")
        return get_rule_based_brief(attack_type)

    # Check cache first
    key = _cache_key(attack_type, details)
    if key in _cache:
        logger.info(f"Cache hit for {attack_type}")
        cached = _cache[key].copy()
        cached['source'] = 'cache'
        return cached

    # Build prompt
    template = PROMPT_TEMPLATES.get(attack_type, PROMPT_TEMPLATES['default'])
    prompt   = template.format(
        attack_type=attack_type,
        details=json.dumps(details),
        **{k: details.get(k, 'N/A') for k in details}
    )

    try:
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=300,
            temperature=0.3
        )
        brief_text = response.choices[0].message.content.strip()

        result = {
            'attack_type': attack_type,
            'timestamp':   datetime.now().isoformat(),
            'source':      'groq-llama3',
            'brief':       brief_text
        }

        # Store in cache
        _cache[key] = result
        logger.info(f"Groq brief generated for {attack_type}")
        return result

    except Exception as e:
        logger.warning(f"Groq failed: {e} — using rule-based fallback")
        return get_rule_based_brief(attack_type)


def analyze_batch(attack_counts: dict) -> list:
    """Analyze all attacks found in a batch."""
    briefs = []
    for attack_type, count in attack_counts.items():
        if attack_type == 'BENIGN':
            continue
        details = {
            'count':         count,
            'flows_per_sec': count * 100,
            'packet_rate':   count * 50,
            'source_ips':    f'{min(count, 5)} unique IPs',
            'duration':      10,
            'ports_scanned': count * 3,
            'scan_rate':     count,
            'source_ip':     '192.168.1.100'
        }
        brief = get_groq_brief(attack_type, details)
        briefs.append(brief)
        print(f"\n{'='*60}")
        print(f"🚨 ATTACK: {attack_type} ({count} flows)")
        print(f"📋 SOURCE: {brief['source'].upper()}")
        print(f"📝 BRIEF:\n{brief['brief']}")

    return briefs


if __name__ == '__main__':
    logger.info("Testing AI Threat Analyst with Groq...")
    sample_attacks = {
        'DDoS':       455,
        'PortScan':   561,
        'DoS Hulk':   815,
        'Bot':        4,
        'Heartbleed': 1
    }
    briefs = analyze_batch(sample_attacks)
    logger.info(f"\nTotal briefs generated: {len(briefs)}")

    # Test cache
    logger.info("Testing cache — running same attack again...")
    analyze_batch({'DDoS': 455})