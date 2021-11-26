import numpy as np
import taichi as ti
import time



@ti.data_oriented
class Cloth:
    def __init__(self, N):
        self.N = N
        self.NF = 2 * N**2  # number of faces
        self.NV = (N + 1)**2  # number of vertices
        self.NE = 2 * N * (N + 1) + 2 * N * N  # numbser of edges
        self.initPos = ti.Vector.field(2, ti.f32, self.NV)
        self.pos = ti.Vector.field(2, ti.f32, self.NV)
        self.vel = ti.Vector.field(2, ti.f32, self.NV)
        self.force = ti.Vector.field(2, ti.f32, self.NV)
        self.mass = ti.field(ti.f32, self.NV)
        self.spring = ti.Vector.field(2, ti.i32, self.NE)
        self.rest_len = ti.field(ti.f32, self.NE)
        self.ks = 1000.0  # spring stiffness
        self.kf = 1.0e5  # Attachment point stiffness
        self.Jx = ti.Matrix.field(2, 2, ti.f32, self.NE)  # Force Jacobian
        self.Jf = ti.Matrix.field(2, 2, ti.f32, 2)  # Attachment Jacobian

        self.init_pos()
        self.init_edges()

        # For sparse matrix solver， PPT: P45
        max_num_triplets = 10000
        self.MBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV,
                                                      max_num_triplets)
        self.init_mass_sp(self.MBuilder)
        self.M = self.MBuilder.build()
        self.KBuilder = ti.linalg.SparseMatrixBuilder(2 * self.NV, 2 * self.NV,
                                                      max_num_triplets)

        # For conjugate gradient method, PPT: P106
        self.x = ti.Vector.field(2, ti.f32, self.NV)
        self.Ax = ti.Vector.field(2, ti.f32, self.NV)
        self.b = ti.Vector.field(2, ti.f32, self.NV)
        self.r = ti.Vector.field(2, ti.f32, self.NV)
        self.d = ti.Vector.field(2, ti.f32, self.NV)
        self.Ad = ti.Vector.field(2, ti.f32, self.NV)

    @ti.kernel
    def init_pos(self):
        for i, j in ti.ndrange(self.N + 1, self.N + 1):
            k = i * (self.N + 1) + j
            self.initPos[k] = ti.Vector([i*2, j]) / self.N * 0.3 + ti.Vector(
                [0.2, 0.2])
            self.pos[k] = self.initPos[k]
            self.vel[k] = ti.Vector([0, 0])
            self.mass[k] = 1.0

    @ti.kernel
    def init_edges(self):
        pos, spring, N, rest_len = ti.static(self.pos, self.spring, self.N,
                                             self.rest_len)
        for i, j in ti.ndrange(N + 1, N):
            idx, idx1 = i * N + j, i * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx1 + 1])
        start = N * (N + 1)
        for i, j in ti.ndrange(N, N + 1):
            idx, idx1, idx2 = start + i + j * N, i * (N + 1) + j, i * (
                N + 1) + j + N + 1
            spring[idx] = ti.Vector([idx1, idx2])
        start = 2 * N * (N + 1)
        for i, j in ti.ndrange(N, N):
            idx, idx1, idx2 = start + i * N + j, i * (N + 1) + j, (i + 1) * (
                N + 1) + j + 1
            spring[idx] = ti.Vector([idx1, idx2])
        start = 2 * N * (N + 1) + N * N
        for i, j in ti.ndrange(N, N):
            idx, idx1, idx2 = start + i * N + j, i * (N + 1) + j + 1, (
                i + 1) * (N + 1) + j
            spring[idx] = ti.Vector([idx1, idx2])
        for i in range(self.NE):
            idx1, idx2 = spring[i]
            rest_len[i] = (pos[idx1] - pos[idx2]).norm()

    @ti.kernel
    def init_mass_sp(self, M: ti.linalg.sparse_matrix_builder()):
        for i in range(self.NV):
            M[2 * i + 0, 2 * i + 0] += self.mass[i]
            M[2 * i + 1, 2 * i + 1] += self.mass[i]

    @ti.func
    def clear_force(self):
        for i in self.force:
            self.force[i] = ti.Vector([0.0, 0.0])


    @ti.func
    def reflect(self, v, normal):
        if v.dot(normal)>0:
            normal = -normal
        return v - 2 * v.dot(normal) * normal
            

    @ti.kernel
    def compute_force(self):
        self.clear_force()
        gravity = ti.Vector([0.0, -2.0])
        
        my_vel_new = my_vel[None]
        is_collided[None] = False
        
        for i in self.force:
            self.force[i] += gravity * self.mass[i]
            
            diff = my_pos[None] - self.pos[i]
            diff_norm = diff.norm(min_diff)
            if diff_norm <= radius/res and diff.dot(my_vel[None])<0:
                is_collided[None] = True
                my_force = my_vel[None].dot(diff)/min((diff_norm), 0.001)*0.9
                self.force[i] += my_force*h*500
                my_vel_new = my_vel[None] * -(1-1e-1)        

        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dis = pos1 - pos2
            # Hook's law
            force = self.ks * (dis.norm() -
                               self.rest_len[i]) * dis.normalized()
            self.force[idx1] -= force
            self.force[idx2] += force

            
            dpos = pos1 - pos2
            dpos_norm = dpos.norm(min_diff)
            dpos_n = ti.Vector([dpos[1], -dpos[0]])/dpos_norm
            diff = my_pos[None] - (pos1+pos2)/2
            diff_norm = diff.norm(min_diff)
            if diff_norm <= radius/res and diff.dot(my_vel[None])<0:
                is_collided[None] = True
                my_force = max(my_vel[None].dot(diff)/(diff_norm), 10)*0.9
                self.force[idx1] += my_force*h*1000 - dpos/dpos_norm*h*1000
                self.force[idx2] += my_force*h*1000 + dpos/dpos_norm*h*1000
                my_vel_new = self.reflect(my_vel[None], dpos_n) * (1-1e-1)
        
        if is_collided[None]:
            my_vel[None] = my_vel_new    
            
        # Attachment constraint force
        self.force[self.N] += self.kf * (self.initPos[self.N] -
                                         self.pos[self.N])
        self.force[self.NV - 1] += self.kf * (self.initPos[self.NV - 1] -
                                              self.pos[self.NV - 1])

    @ti.kernel
    def compute_force_Jacobians(self):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            pos1, pos2 = self.pos[idx1], self.pos[idx2]
            dx = pos1 - pos2
            I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            dxtdx = ti.Matrix([[dx[0] * dx[0], dx[0] * dx[1]],
                               [dx[1] * dx[0], dx[1] * dx[1]]])
            l = dx.norm()
            if l != 0.0:
                l = 1.0 / l
            self.Jx[i] = (I - self.rest_len[i] * l *
                          (I - dxtdx * l**2)) * self.ks
        # Attachment constraint force Jacobian
        self.Jf[0] = ti.Matrix([[-self.kf, 0], [0, -self.kf]])
        self.Jf[1] = ti.Matrix([[-self.kf, 0], [0, -self.kf]])

    @ti.kernel
    def assemble_K(self, K: ti.linalg.sparse_matrix_builder()):
        for i in self.spring:
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            for m, n in ti.static(ti.ndrange(2, 2)):
                K[2 * idx1 + m, 2 * idx1 + n] -= self.Jx[i][m, n]
                K[2 * idx1 + m, 2 * idx2 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx1 + n] += self.Jx[i][m, n]
                K[2 * idx2 + m, 2 * idx2 + n] -= self.Jx[i][m, n]
        for m, n in ti.static(ti.ndrange(2, 2)):
            K[2 * self.N + m, 2 * self.N + n] += self.Jf[0][m, n]
            K[2 * (self.NV - 1) + m, 2 * (self.NV - 1) + n] += self.Jf[1][m, n]

    @ti.kernel
    def directUpdatePosVel(self, h: ti.f32, v_next: ti.ext_arr()):
        for i in self.pos:
            self.vel[i] = ti.Vector([v_next[2 * i], v_next[2 * i + 1]])
            self.pos[i] += h * self.vel[i]

    def update_direct(self, h):
        self.compute_force()
        self.compute_force_Jacobians()
        # Assemble global system
        self.assemble_K(self.KBuilder)
        K = self.KBuilder.build()
        A = self.M - h**2 * K
        solver = ti.linalg.SparseSolver(solver_type="LLT")
        solver.analyze_pattern(A)
        solver.factorize(A)

        vel = self.vel.to_numpy().reshape(2 * self.NV)
        force = self.force.to_numpy().reshape(2 * self.NV)
        b = h * force + self.M @ vel

        v_next = solver.solve(b)
        # flag = solver.info()
        # print("solver flag: ", flag)
        self.directUpdatePosVel(h, v_next)

    @ti.kernel
    def cgUpdatePosVel(self, h: ti.f32):
        for i in self.pos:
            self.vel[i] = self.x[i]
            self.pos[i] += h * self.vel[i]

    @ti.kernel
    def compute_RHS(self, h: ti.f32):
        #rhs = b = h * force + M @ v
        for i in range(self.NV):
            self.b[i] = h * self.force[i] + self.mass[i] * self.vel[i]

    @ti.func
    def dot(self, v1, v2):
        result = 0.0
        for i in range(self.NV):
            result += v1[i][0] * v2[i][0]
            result += v1[i][1] * v2[i][1]
        return result

    @ti.func
    def A_mult_x(self, h, dst, src):
        coeff = -h**2
        for i in range(self.NV):
            dst[i] = self.mass[i] * src[i]
        for i in range(self.NE):
            idx1, idx2 = self.spring[i][0], self.spring[i][1]
            temp = self.Jx[i] @ (src[idx1] - src[idx2])
            dst[idx1] -= coeff * temp
            dst[idx2] += coeff * temp
        # Attachment constraint
        Attachment1, Attachment2 = self.N, self.NV - 1
        dst[Attachment1] -= coeff * self.kf * src[Attachment1]
        dst[Attachment2] -= coeff * self.kf * src[Attachment2]

    # conjugate gradient solving
    # https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    @ti.kernel
    def before_ite(self) -> ti.f32:
        for i in range(self.NV):
            self.x[i] = ti.Vector([0.0, 0.0])
        self.A_mult_x(h, self.Ax, self.x)  # Ax = A @ x
        for i in range(self.NV):  # r = b - A @ x
            self.r[i] = self.b[i] - self.Ax[i]
        for i in range(self.NV):  # d = r
            self.d[i] = self.r[i]
        delta_new = self.dot(self.r, self.r)
        return delta_new

    @ti.kernel
    def run_iteration(self, delta_new: ti.f32) -> ti.f32:
        self.A_mult_x(h, self.Ad, self.d)  # Ad = A @ d
        alpha = delta_new / self.dot(self.d,
                                     self.Ad)  # alpha = (r^T * r) / dot(d, Ad)
        for i in range(self.NV):
            self.x[i] += alpha * self.d[i]  # x^{i+1} = x^{i} + alpha * d
            self.r[i] -= alpha * self.Ad[i]  # r^{i+1} = r^{i} + alpha * Ad
        delta_old = delta_new
        delta_new = self.dot(self.r, self.r)
        beta = delta_new / delta_old
        for i in range(self.NV):
            self.d[i] = self.r[i] + beta * self.d[
                i]  #p^{i+1} = r^{i+1} + beta * p^{i}
        return delta_new

    def cg(self, h: ti.f32):
        delta_new = self.before_ite()
        ite, iteMax = 0, 2 * self.NV
        while ite < iteMax and delta_new > 1.0e-6:
            delta_new = self.run_iteration(delta_new)
            ite += 1

    def update_cg(self, h):
        self.compute_force()
        self.compute_force_Jacobians()
        self.compute_RHS(h)
        self.cg(h)
        self.cgUpdatePosVel(h)

    def display(self, gui, radius=5, color=0xffffff):
        springs, pos = self.spring.to_numpy(), self.pos.to_numpy()
        line_Begin = np.zeros(shape=(springs.shape[0], 2))
        line_End = np.zeros(shape=(springs.shape[0], 2))
        for i in range(springs.shape[0]):
            idx1, idx2 = springs[i][0], springs[i][1]
            line_Begin[i], line_End[i] = pos[idx1], pos[idx2]
        gui.lines(line_Begin, line_End, radius=2, color=0x0000ff)
        gui.circles(self.pos.to_numpy(), radius, color)


def move():
    vx , vy = my_vel[None][0], my_vel[None][1]
    x , y = my_pos[None][0], my_pos[None][1]
    
    # if not is_collided[None]:
    if gui.is_pressed('w'): 
        if vy>v_boot: vy += acc_y
        else: vy = v_boot*3
    else:
        vy += -h
    if gui.is_pressed('s'): vy -= acc_y if vy< v_boot else acc_y*3
    if gui.is_pressed('a'): vx -= acc_x if vx< v_boot else acc_x*3
    if gui.is_pressed('d'): vx += acc_x if vy>-v_boot else acc_x*3

    
    x += vx*h
    y += vy*h

    #floor resist
    if y<=bottom:
        vx*=min(1, (1-1e-2)*max(-vy, h)/h)


    if x>1: x=1; vx*=-0.7
    if x<0: x=0; vx*=-0.7
    if y>1: y=1; vy*=-0.7
    if y<bottom: 
        y=bottom
        gui.get_event(ti.GUI.PRESS)
        if gui.is_pressed('s'):
            vy*=-0.5
            if abs(vy)<v_boot:
                vy=0
        else:
            vy*=-0.7
    
    
    my_vel[None][0], my_vel[None][1] = vx,vy
    my_pos[None][0], my_pos[None][1] = x,y
    



ti.init(arch=ti.cpu)
cloth = Cloth(N=5)
use_cg = True

res = 800
gui = ti.GUI('Mass Spring System with Ball', res=(res, res))
pause = False
h, max_step = 0.01, 3


acc_x = 0.01/max_step
acc_y = acc_x
v_boot = 0.1

radius = 30
bottom = radius/res
min_diff = 1e-5

my_pos = ti.Vector.field(2, ti.f32, ())
my_vel = ti.Vector.field(2, ti.f32, ())
my_pos[None] = ti.Vector([0.5, 1])
my_vel[None] = ti.Vector([0, 0])
is_collided= ti.field(ti.i32, ())
is_collided[None]=False

# tick=0
while gui.running:
    gui.get_event(ti.GUI.PRESS)
    # for e in gui.get_events(ti.GUI.PRESS):
        # if e.key == gui.ESCAPE:
            # gui.running = False
        # elif e.key == gui.SPACE:
            # pause = not pause
    if not pause:
        for i in range(max_step):
            move()
            
            cloth.update_cg(h)
            # cloth.update_direct(h)

    cloth.display(gui)
    gui.circle(my_pos[None].to_numpy(), color=0xff0000, radius=radius)
    gui.show()
    # tick+=1
    # gui.show(f'img/{tick:0>3d}.png')
