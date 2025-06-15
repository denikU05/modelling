import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt

guiFlag = True

dt = 1/240
th0 = 0.5
g = 10
L = 0.8
L1 = L
L2 = L
m = 1

# Заданные положения джоинтов (радианы)
th1_desired = 1.0
th2_desired = 0.7

# ПИД коэффициенты для каждого джоинта
# Джоинт 1
kp1 = 150.0
ki1 = 40.0
kd1 = 30.0

# Джоинт 2
kp2 = 120.0
ki2 = 20.0
kd2 = 25.0

# Ограничения для интегратора (anti-windup)
integral_limit_1 = 10.0
integral_limit_2 = 10.0

# Переменные для ПИД-регулятора
error_integral_1 = 0.0
error_integral_2 = 0.0
prev_error_1 = 0.0
prev_error_2 = 0.0

physicsClient = p.connect(p.GUI if guiFlag else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-g)
boxId = p.loadURDF("./two-link.urdf.xml", useFixedBase=True)

# Убираем демпфирование
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# Переходим в начальное положение
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=th0, controlMode=p.POSITION_CONTROL)
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=3, targetPosition=0.0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# Отключаем моторы для свободного движения
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=3, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

maxTime = 4
logTime = np.arange(0, maxTime, dt)
sz = len(logTime)

# Логирование
logTh1 = np.zeros(sz)
logTh2 = np.zeros(sz)
logTh1_desired = np.zeros(sz)
logTh2_desired = np.zeros(sz)
logTorque1 = np.zeros(sz)
logTorque2 = np.zeros(sz)
logError1 = np.zeros(sz)
logError2 = np.zeros(sz)

idx = 0

print(f"Целевое положение: th1 = {th1_desired:.3f} рад, th2 = {th2_desired:.3f} рад")

for t in logTime:
    # Получаем текущие положения и скорости джоинтов
    th1_current = p.getJointState(boxId, 1)[0]
    vel1_current = p.getJointState(boxId, 1)[1]
    th2_current = p.getJointState(boxId, 3)[0]
    vel2_current = p.getJointState(boxId, 3)[1]

    # Вычисляем ошибки
    error_1 = th1_desired - th1_current
    error_2 = th2_desired - th2_current

    # Интегральная составляющая с защитой от насыщения (anti-windup)
    error_integral_1 += error_1 * dt
    error_integral_2 += error_2 * dt

    # Ограничиваем интегральную составляющую
    error_integral_1 = np.clip(error_integral_1, -integral_limit_1, integral_limit_1)
    error_integral_2 = np.clip(error_integral_2, -integral_limit_2, integral_limit_2)

    # Дифференциальная составляющая с фильтрацией
    if idx > 0:  # избегаем деления на ноль в первой итерации
        error_derivative_1 = (error_1 - prev_error_1) / dt
        error_derivative_2 = (error_2 - prev_error_2) / dt
    else:
        error_derivative_1 = 0.0
        error_derivative_2 = 0.0

    # ПИД управление
    torque_1 = kp1 * error_1 + ki1 * error_integral_1 + kd1 * error_derivative_1
    torque_2 = kp2 * error_2 + ki2 * error_integral_2 + kd2 * error_derivative_2

    # Ограичиваем максимальный момент
    max_torque = 150.0  # увеличен лимит момента
    torque_1 = np.clip(torque_1, -max_torque, max_torque)
    torque_2 = np.clip(torque_2, -max_torque, max_torque)

    # Применяем моменты к джоинтам
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1,
                            controlMode=p.TORQUE_CONTROL, force=torque_1)
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=3,
                            controlMode=p.TORQUE_CONTROL, force=torque_2)

    # Логирование
    logTh1[idx] = th1_current
    logTh2[idx] = th2_current
    logTh1_desired[idx] = th1_desired
    logTh2_desired[idx] = th2_desired
    logTorque1[idx] = torque_1
    logTorque2[idx] = torque_2
    logError1[idx] = error_1
    logError2[idx] = error_2

    # Сохраняем предыдущие ошибки
    prev_error_1 = error_1
    prev_error_2 = error_2

    p.stepSimulation()
    idx += 1

    if guiFlag:
        time.sleep(dt)

    # Выводим прогресс каждые 0.5 секунды
    if idx % int(0.5/dt) == 0:
        print(f"t={t:.2f}s: th1={th1_current:.3f} (цель: {th1_desired:.3f}), "
              f"th2={th2_current:.3f} (цель: {th2_desired:.3f})")

p.disconnect()

# Построение графиков
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Положения джоинтов
axes[0,0].plot(logTime, logTh1, 'b-', label='Текущее положение th1')
axes[0,0].plot(logTime, logTh1_desired, 'r--', label='Желаемое положение th1')
axes[0,0].set_ylabel('Угол (рад)')
axes[0,0].set_title('Джоинт 1')
axes[0,0].legend()
axes[0,0].grid(True)

axes[0,1].plot(logTime, logTh2, 'b-', label='Текущее положение th2')
axes[0,1].plot(logTime, logTh2_desired, 'r--', label='Желаемое положение th2')
axes[0,1].set_ylabel('Угол (рад)')
axes[0,1].set_title('Джоинт 2')
axes[0,1].legend()
axes[0,1].grid(True)

# Ошибки
axes[1,0].plot(logTime, logError1, 'g-')
axes[1,0].set_ylabel('Ошибка (рад)')
axes[1,0].set_title('Ошибка Джоинт 1')
axes[1,0].grid(True)

axes[1,1].plot(logTime, logError2, 'g-')
axes[1,1].set_ylabel('Ошибка (рад)')
axes[1,1].set_title('Ошибка Джоинт 2')
axes[1,1].grid(True)

# Моменты
axes[2,0].plot(logTime, logTorque1, 'm-')
axes[2,0].set_ylabel('Момент (Н·м)')
axes[2,0].set_xlabel('Время (с)')
axes[2,0].set_title('Управляющий момент Джоинт 1')
axes[2,0].grid(True)

axes[2,1].plot(logTime, logTorque2, 'm-')
axes[2,1].set_ylabel('Момент (Н·м)')
axes[2,1].set_xlabel('Время (с)')
axes[2,1].set_title('Управляющий момент Джоинт 2')
axes[2,1].grid(True)

plt.tight_layout()
plt.show()

# Финальная статистика
final_error_1 = abs(logError1[-1])
final_error_2 = abs(logError2[-1])
print(f"\nФинальные ошибки:")
print(f"Джоинт 1: {final_error_1:.4f} рад ({np.degrees(final_error_1):.2f}°)")
print(f"Джоинт 2: {final_error_2:.4f} рад ({np.degrees(final_error_2):.2f}°)")