from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, WritePrecision
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

# InfluxDB config
INFLUX_TOKEN = "your_token"
INFLUX_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUX_ORG = "your_org"
INFLUX_BUCKET = "your_bucket"

@app.route('/convert_and_upload', methods=['POST'])
def convert_and_upload():
    try:
        file = request.files.get('file')
        if file is None:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))

        today = datetime.now().date()
        start_dt = datetime.combine(today, datetime.min.time())

        with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
            write_api = client.write_api(write_options=WritePrecision.S)

            for _, row in df.iterrows():
                try:
                    minutes, seconds = map(int, row["Timestamp"].split(":"))
                    delta = timedelta(minutes=minutes, seconds=seconds)
                    time_dt = start_dt + delta
                except:
                    continue

                is_dangerous = 1 if str(row["Is Dangerous"]).strip().upper() == "TRUE" else 0
                emotion = str(row["Emotion"]).strip()

                point = (
                    Point("interview")
                    .tag("emotion", emotion)
                    .field("is_dangerous", is_dangerous)
                    .time(time_dt, WritePrecision.S)
                )
                write_api.write(bucket=INFLUX_BUCKET, record=point)

        return jsonify({"message": "Data successfully uploaded to InfluxDB"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)