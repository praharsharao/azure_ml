
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def main():
    ml_client = MLClient.from_config(DefaultAzureCredential())
    
    print("🔍 Scanning for stale endpoints...")
    # List all endpoints
    endpoints = ml_client.online_endpoints.list()
    
    # Target anything starting with 'insurance'
    targets = [e for e in endpoints if e.name.startswith("insurance")]
    
    if not targets:
        print("✅ No stale endpoints found. Your quota should be free.")
        return

    print(f"⚠️ Found {len(targets)} stale/failed endpoints. Deleting them to free quota...")
    
    for t in targets:
        print(f"🗑️ Deleting {t.name} (This takes 2-3 mins)...")
        try:
            # begin_delete is async, we wait() to ensure it's gone
            ml_client.online_endpoints.begin_delete(name=t.name).wait()
            print("   Deleted.")
        except Exception as e:
            print(f"   Error deleting: {e}")

    print("\n✅ Cleanup complete! You are ready to deploy.")

if __name__ == "__main__":
    main()