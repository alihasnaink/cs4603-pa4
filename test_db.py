from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv()
w = WorkspaceClient()
print(w.current_user.me().user_name)
