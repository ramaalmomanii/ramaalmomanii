import numpy as np
import random
import matplotlib.pyplot as plt

num_user = 3
num_particles = 5
itar = 100

# Objective function
def engagement_score(schedule, w1=0.4, w2=0.3, w3=0.2, w4=0.1):
    schedule = np.clip(schedule, 8, 22)
   
    # توليد قيم عشوائية للتفاعلات
    likes = np.random.randint(0, 100, size=len(schedule))
    shares = np.random.randint(0, 100, size=len(schedule))
    comments = np.random.randint(0, 100, size=len(schedule))
    time_spent = np.random.uniform(0, 100, size=len(schedule))  
    score = w1 * likes + w2 * shares + w3 * comments + w4 * time_spent
  
    return likes, shares, comments, time_spent, score.sum()

# Particle Swarm Optimization function 
w = 0.5 
c1 = 1.5 
c2 = 1.5
particles = []

# Initialize particles
for i in range(num_particles):  
    particle = {
        'position': [random.uniform(8, 22) for _ in range(num_user)], 
        'velocity': [random.uniform(-1, 1) for _ in range(num_user)],    
        'best_position': None,  
        'best_value': float('inf'),  
    }
    particles.append(particle)

gp_best = None
gv_best = float('inf')

# Save the initial best particle before optimization
initial_best_particle_index = None
initial_best_total_score = float('-inf')

for i, particle in enumerate(particles):
    total_score = sum(engagement_score(particle['position'])[4] for _ in range(num_user))
    if total_score > initial_best_total_score:
        initial_best_total_score = total_score
        initial_best_particle_index = i

initial_best_particle = particles[initial_best_particle_index]
initial_best_particle_position = initial_best_particle['position']

# PSO iterations
for iteration in range(itar):
    for particle in particles:
        current_value = sum(engagement_score(particle['position'])[4] for _ in range(num_user))
        # Update personal best
        if current_value < particle['best_value']:
            particle['best_position'] = particle['position'].copy()
            particle['best_value'] = current_value
        # Update global best
        if current_value < gv_best:
            gp_best = particle['position'].copy()
            gv_best = current_value
        
        # Update particle velocity
        for d in range(num_user):
            r1 = random.random()
            r2 = random.random()
            particle['velocity'][d] = (
                w * particle['velocity'][d] +
                c1 * r1 * (particle['best_position'][d] - particle['position'][d]) +
                c2 * r2 * (gp_best[d] - particle['position'][d])
            )
        
        # Update particle position
        for d in range(num_user):
            particle['position'][d] += particle['velocity'][d]

# Print the final information for all particles
best_particle_index = -1
best_total_score = float('-inf')

for i, particle in enumerate(particles):
    print(f"Particle {i + 1}:")
    total_score = 0
    for u in range(num_user):
        likes, shares, comments, time_spent, score = engagement_score(particle['position'])
        total_score += score
        print(f"  User {u + 1}:")
        print(f"    Likes: {likes.sum()}, Shares: {shares.sum()}, Comments: {comments.sum()}, Time Spent: {time_spent.sum():.1f} min")
        print(f"    User {u + 1} Score: {score:.2f}")
    print(f"  Total Engagement Score for Particle {i + 1}: {total_score:.2f}")
    print('-' * 60)

    # Track the particle with the highest total score
    if total_score > best_total_score:
        best_total_score = total_score
        best_particle_index = i

# Print the best particle
print(f"Best Particle ({best_particle_index + 1}):")
best_particle = particles[best_particle_index]
for u in range(num_user):
    likes, shares, comments, time_spent, score = engagement_score(best_particle['position'])
    print(f"  User {u + 1}:")
    print(f"    Likes: {likes.sum()}, Shares: {shares.sum()}, Comments: {comments.sum()}, Time Spent: {time_spent.sum():.1f} min")
    print(f"    User {u + 1} Score: {score:.2f}")
print(f"  Total Engagement Score for Best Particle: {best_total_score:.2f}")

# Plotting the initial best particle's user data
schedule_times = np.linspace(8, 22, num=len(initial_best_particle_position))

for u in range(num_user):
    likes, shares, comments, time_spent, _ = engagement_score(initial_best_particle_position)
    plt.figure(figsize=(12, 6))
    plt.plot(schedule_times, likes, label='Likes (Initial)', marker='o')
    plt.plot(schedule_times, shares, label='Shares (Initial)', marker='o')
    plt.plot(schedule_times, comments, label='Comments (Initial)', marker='o')
    plt.plot(schedule_times, time_spent, label='Time Spent (Initial)', marker='o')
    plt.title(f'User {u + 1} Engagement Metrics for Initial Best Particle')
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.show()
input("#ENTER TO SHOW EHE >> Plotting the final best particle's user data ")

# Plotting the final best particle's user data
final_best_particle_position = particles[best_particle_index]['position']

for u in range(num_user):
    likes, shares, comments, time_spent, _ = engagement_score(final_best_particle_position)
    plt.figure(figsize=(12, 6))
    plt.plot(schedule_times, likes, label='Likes (Final)', marker='o')
    plt.plot(schedule_times, shares, label='Shares (Final)', marker='o')
    plt.plot(schedule_times, comments, label='Comments (Final)', marker='o')
    plt.plot(schedule_times, time_spent, label='Time Spent (Final)', marker='o')
    plt.title(f'User {u + 1} Engagement Metrics for Final Best Particle')
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.show()


