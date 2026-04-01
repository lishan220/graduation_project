from flask import Flask, request, jsonify

app = Flask(__name__)

# 模拟数据库
users = {}
next_id = 1

@app.route('/users', methods=['POST'])
def create_user():
    global next_id
    data = request.get_json()
    username = data.get('username')
    age = data.get('age')

    # 业务逻辑：校验
    if not username or len(username) < 3 or len(username) > 20:
        return jsonify({"error": "用户名长度必须为3-20字符"}), 400
    if age is not None:
        if not isinstance(age, int) or age < 0 or age > 120:
            return jsonify({"error": "年龄必须在0-120之间"}), 400

    user_id = next_id
    users[user_id] = {"username": username, "age": age}
    next_id += 1
    return jsonify({"id": user_id, "username": username, "age": age}), 201

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = users.get(user_id)
    if not user:
        return jsonify({"error": "用户不存在"}), 404
    return jsonify({"id": user_id, **user}), 200

@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    if user_id in users:
        del users[user_id]
        return '', 204
    return jsonify({"error": "用户不存在"}), 404

@app.route('/users', methods=['GET'])
def list_users():
    return jsonify([{"id": uid, **data} for uid, data in users.items()]), 200

if __name__ == '__main__':
    # 预置一个用户
    users[1] = {"username": "preloaded", "age": 30}
    app.run(debug=True, port=5000)