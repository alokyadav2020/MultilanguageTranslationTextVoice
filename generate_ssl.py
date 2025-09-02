# # generate_ssl.py
# from cryptography import x509
# from cryptography.x509.oid import NameOID
# from cryptography.hazmat.primitives import hashes, serialization
# from cryptography.hazmat.primitives.asymmetric import rsa
# import datetime
# import ipaddress

# def generate_ssl_certificates(common_name="localhost", output_dir="."):
#     """Generate SSL certificates using Python cryptography library"""
    
#     print(f"🔐 Generating SSL certificates for {common_name}...")
    
#     # Generate private key
#     print("🔑 Generating private key...")
#     private_key = rsa.generate_private_key(
#         public_exponent=65537,
#         key_size=2048,
#     )
    
#     # Create certificate
#     print("📜 Creating certificate...")
#     subject = issuer = x509.Name([
#         x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
#         x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
#         x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
#         x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Organization"),
#         x509.NameAttribute(NameOID.COMMON_NAME, common_name),
#     ])
    
#     # Build certificate
#     cert = x509.CertificateBuilder().subject_name(
#         subject
#     ).issuer_name(
#         issuer
#     ).public_key(
#         private_key.public_key()
#     ).serial_number(
#         x509.random_serial_number()
#     ).not_valid_before(
#         datetime.datetime.utcnow()
#     ).not_valid_after(
#         datetime.datetime.utcnow() + datetime.timedelta(days=365)
#     ).add_extension(
#         x509.SubjectAlternativeName([
#             x509.DNSName("localhost"),
#             x509.DNSName("*.localhost"),
#             x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
#             x509.IPAddress(ipaddress.ip_address("::1")),
#         ]),
#         critical=False,
#     ).add_extension(
#         x509.KeyUsage(
#             digital_signature=True,
#             key_encipherment=True,
#             data_encipherment=False,
#             key_agreement=False,
#             key_cert_sign=False,
#             crl_sign=False,
#             content_commitment=False,
#             encipher_only=False,
#             decipher_only=False,
#         ),
#         critical=True,
#     ).add_extension(
#         x509.ExtendedKeyUsage([
#             x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
#         ]),
#         critical=True,
#     ).sign(private_key, hashes.SHA256())
    
#     # Write private key
#     key_path = f"{output_dir}/key.pem"
#     with open(key_path, "wb") as f:
#         f.write(private_key.private_bytes(
#             encoding=serialization.Encoding.PEM,
#             format=serialization.PrivateFormat.PKCS8,
#             encryption_algorithm=serialization.NoEncryption()
#         ))
    
#     # Write certificate
#     cert_path = f"{output_dir}/cert.pem"
#     with open(cert_path, "wb") as f:
#         f.write(cert.public_bytes(serialization.Encoding.PEM))
    
#     print("✅ SSL certificates generated successfully!")
#     print(f"📁 Files created:")
#     print(f"   🔑 Private Key: {key_path}")
#     print(f"   📜 Certificate: {cert_path}")
    
#     return key_path, cert_path

# if __name__ == "__main__":
#     # Install required library
#     import subprocess
#     import sys
    
#     try:
#         import cryptography
#     except ImportError:
#         print("📦 Installing cryptography library...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography"])
#         import cryptography
    
#     # Generate certificates
#     generate_ssl_certificates()

import uvicorn
import ssl
import os
import subprocess
from pathlib import Path

def generate_ssl_cert():
    """Generate self-signed SSL certificate for development"""
    cert_path = Path("cert.pem")
    key_path = Path("key.pem")
    
    if not cert_path.exists() or not key_path.exists():
        print("🔧 Generating SSL certificates...")
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:4096",
            "-keyout", "key.pem", "-out", "cert.pem", "-days", "365", "-nodes",
            "-subj", "/C=US/ST=Development/L=Local/O=Dev/CN=192.168.1.39"
        ]
        try:
            subprocess.run(cmd, check=True)
            print("✅ SSL certificates generated successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to generate certificates. Please install OpenSSL.")
            return False
        except FileNotFoundError:
            print("❌ OpenSSL not found. Please install OpenSSL first.")
            return False
    return True

def start_https_server():
    """Start the FastAPI server with HTTPS"""
    if not generate_ssl_cert():
        print("⚠️  Starting without HTTPS...")
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
        return
    
    print("🔒 Starting HTTPS server...")
    print("📱 Access your app at: https://192.168.1.39:8000")
    print("🎤 Voice features will be available!")
    print("⚠️  You may need to accept the security warning in your browser")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
        ssl_version=ssl.PROTOCOL_TLS,
        ssl_cert_reqs=ssl.CERT_NONE
    )

if __name__ == "__main__":
    start_https_server()