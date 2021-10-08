const SIM_SIZE = 1;

const MAX_SIZE = 0.02;
const MIN_SIZE = 0.002;

const MAX_SPLIT_THRESH = 100;
const MIN_SPLIT_THRESH = 0.05

const MIN_POP = 50;
const MAX_POP = 2000;
const MAX_INIT_VEL = 0.0001;
const VEL_JITTER = MAX_INIT_VEL;
const POS_JITTER = (MAX_SIZE + MIN_SIZE) / 2;
const ANG_MOVE_SPEED = 0.05;
const LIN_MOVE_SPEED = MAX_INIT_VEL;

const EAT_SIZE_THRESH = 0.95;
const ENERGY_SIZE_COST = 0.12 / (MAX_SIZE * MAX_SIZE);
const ENERGY_MOVE_COST = 0.0005;
const ENERGY_OVERLAP_COST = 0.005;
const BASE_ENERGY_INCREASE = ENERGY_SIZE_COST * ((MAX_SIZE - MIN_SIZE) / 10 + MIN_SIZE)**2;

const FRICTION_COEFF = 0.99;

console.log(BASE_ENERGY_INCREASE);

const MUT_PROB = 0.5;
const SIZE_MUT_FRAC = 0.02;
const SPLIT_R_MUT_SIZE = 0.01;
const SPLIT_T_MUT_SIZE = 0.1;
const CLOCK_RATE_MUT_SIZE = 0.01;
const COLOR_MUT_SIZE = 5;
const BRAIN_PARAM_MUT_SIZE = 0.01;

const BRAIN_SIZE = 64;

const BRAIN_HID_SIZE = 16;
const BRAIN_LAYERS = 4;
const BRAIN_MEM_SIZE = 8;


const rand_in_range = (low, high) => {
  return Math.random() * (high - low) + low;
}

const rand_color = () => {
  return Math.random().toString(16).slice(2,8);
}

const rand_angle = () => {
  return Math.random() * Math.PI * 2;
}

const rand_hex = (length) => {
  let result = '';
  while (result.length < length) {
    result += Math.random().toString(16).slice(2);
  }

  return result.slice(0, length);
}


const rand_less_bit_hex = (len, n) => {
  let result = parseInt(rand_hex(len), 16);
  for (let i=0; i < n; i++) {
    result = result & parseInt(rand_hex(len), 16);
  }
  return result;
}

const clip = (x, min, max) => {
  return Math.min(Math.max(x,min),max);
}

const invert_color = (color) => {
  const r = parseInt(color.slice(0,2));
  const g = parseInt(color.slice(2,4));
  const b = parseInt(color.slice(4,6));
  
  if ((r + g + b) / 3 < 128) {
    return 'ffffff';
  }
  else {
    return '000000';
  }

}


class Vec2 {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  static rand() {
    return new Vec2(Math.random(), Math.random());
  }

  static rand_unit() {
    return Vec2.from_angle(Math.random() * Math.PI * 2);
  }

  static from_angle(angle) {
    return new Vec2(Math.cos(angle), Math.sin(angle));
  }

  asUnit() {
    return this.mulSc(1 / (this.norm()));
  }

  mulSc(scalar) {
    return new Vec2(this.x * scalar, this.y * scalar);
  }

  addSc(scalar) {
    return new Vec2(this.x + scalar, this.y + scalar);
  }

  mulVec(other) {
    return new Vec2(this.x * other.getX(), this.y * other.getY());
  }

  addVec(other) {
    return new Vec2(this.x + other.getX(), this.y + other.getY());
  }

  dot(other) {
    return this.x * other.getX() + this.y * other.getY();
  }

  cross(other) {
    return this.x*other.getY() - this.y*other.getX();
  }

  rotate(angle) {
    const sin_ang = Math.sin(angle);
    const cos_ang = Math.cos(angle);
    return new Vec2(this.x*cos_ang - this.y * sin_ang, this.x * sin_ang + this.y * cos_ang );
  }

  norm() {
    return Math.sqrt((this.x * this.x) + (this.y * this.y));
  }

  getX() {
    return this.x;
  }

  getY() {
    return this.y;
  }
}


/*
  Basic neural net
 */
class BasicNN {
  constructor(ins, hids, outs, layers, weights, biases) {
    this.weights = weights;
    this.biases = biases;

    this.num_layers = layers;
    this.in_size = ins;
    this.hid_size = hids;
    this.out_size = outs;
  }

  static rand(ins, hids, outs, layers) {
    const weights = [];
    const num_weights = (layers-2)*hids*hids + ins*hids + hids*outs;
    for (let i=0; i < num_weights; i++) {
      weights.push(Math.random()-0.5);
    }

    const biases = [];
    const num_biases = (layers-1)*hids + outs;
    for (let i=0; i < num_biases; i++) {
      biases.push(Math.random() - 0.5);
    }

    return new BasicNN(ins, hids, outs, layers, weights, biases);
  }


  forward(x) {
    
    let layer_x = BasicNN.matmul(x, this.weights, this.in_size, 
                                 this.hid_size, 0);
    layer_x = BasicNN.addVec(layer_x, this.biases, this.hid_size, 0);
    layer_x = BasicNN.hid_act(layer_x, this.hid_size)
    
    let weight_start = this.in_size * this.hid_size; 
    let bias_start = this.hid_size;

    for (let i=1; i < this.num_layers-1; i++) {
      layer_x = BasicNN.matmul(layer_x, this.weights, this.hid_size, 
                               this.hid_size, weight_start);
      layer_x = BasicNN.addVec(layer_x, this.biases, this.hid_size, 
                               bias_start);
      layer_x = BasicNN.hid_act(layer_x, this.hid_size);
      weight_start += this.hid_size * this.hid_size;
      bias_start += this.hid_size;
    }

    layer_x = BasicNN.matmul(layer_x, this.weights, this.hid_size,
                             this.out_size, weight_start);
    layer_x = BasicNN.addVec(layer_x, this.biases, this.out_size,
                             bias_start);
    return BasicNN.out_act(layer_x, this.out_size);
  }


  static matmul(x, W, w_n, w_m, w_index_start) {
    const result = [];
    let sum;
    let W_i;
    for (let i=0; i < w_m; i++) {
      sum = 0;
      W_i = w_index_start + i * w_n;
      for (let j=0; j < w_n; j++) {
        sum += x[j] * W[W_i + j];
      }
      result[i] = sum;
    }
    return result;
  }

  static addVec(x, b, n, b_index_start) {
    const result = [];
    for (let i=0; i < n; i++) {
      result[i] = x[i] + b[i + b_index_start];
    }

    return result;
  }

  static hid_act(x, n) {
    const result = [];
    for (let i=0; i<n; i++) {
      result[i] = Math.max(x[i], 0);
    }
    return result;
  }

  static out_act(x, n) {
    return x;
  }

  getWeights() {
    return this.weights;
  }

  getBiases() {
    return this.biases;
  }

  getInSize() {
    return this.in_size;
  }
  
  getHidSize() {
    return this.hid_size;
  }

  getOutSize() {
    return this.out_size;
  }

  getNumLayers() {
    return this.num_layers;
  }

}





class NeuralBrain {
  constructor(nn) {
    this.clock = 0;
    
    this.memory = new Array(BRAIN_MEM_SIZE).fill(0);
    this.nn = nn;

    this.closest_diff_vec = Vec2.rand();
    this.closest_dist = SIM_SIZE * 2;
    this.closest_size = MIN_SIZE;
    
    this.angle_vec = undefined;
    this.energy = undefined;
    this.size = undefined;
  }

  static rand() {
    return new NeuralBrain(BasicNN.rand(BRAIN_MEM_SIZE + 6, BRAIN_HID_SIZE, BRAIN_MEM_SIZE + 2, BRAIN_LAYERS));
  }


  getMove() {
    //calc closest_angle
    const diff_unit = this.closest_diff_vec.asUnit();
    const angle_cross = this.angle_vec.cross(diff_unit);
    const angle_dot = this.angle_vec.dot(diff_unit);
    const relative_size = this.size / this.closest_size - 1;
    const ins = [
                  this.clock, 
                  this.energy,
                  this.closest_dist,
                  angle_cross,
                  angle_dot,
                  relative_size,
                ];

    for (let i=0; i<this.memory.length; i++){
      ins.push(this.memory[i]);
    }

    const outs = this.nn.forward(ins);
  
    this.memory = outs.slice(0, BRAIN_MEM_SIZE);
    this.clock = (this.clock + 1) % 256;

    return outs.slice(BRAIN_MEM_SIZE, BRAIN_MEM_SIZE+2);
  }

  mut_copy() {
    const weights = this.nn.getWeights();
    const biases = this.nn.getBiases();

    const new_weights = weights.map((x) => {
      return x + rand_in_range(-BRAIN_PARAM_MUT_SIZE, BRAIN_PARAM_MUT_SIZE);
    });

    const new_biases = biases.map((x) => {
      return x + rand_in_range(-BRAIN_PARAM_MUT_SIZE, BRAIN_PARAM_MUT_SIZE);
    });


    const new_nn = new BasicNN(
                               this.nn.getInSize(), 
                               this.nn.getHidSize(), 
                               this.nn.getOutSize(), 
                               this.nn.getNumLayers(), 
                               new_weights, 
                               new_biases
                       );
    return new NeuralBrain(new_nn);
  }

  setEnergy(energy) {
    this.energy = energy;
  }

  setAngleVec(angle_vec) {
    this.angle_vec = angle_vec;
  }

  setSize(size) {
    this.size = size;
  }

  sense(dist, diff_vec, other_size) {
    if (dist < this.closest_dist) {
      this.closest_dist = dist;
      this.closest_diff_vec = diff_vec;
      this.closest_size = other_size;
    }
  }


}


class Brain {
  constructor(program, clock_rate) {
    this.clock = 0;
    this.clock_rate = clock_rate;
    this.program = program;
  }

  static rand() {
    //return new Brain('0'.repeat(BRAIN_SIZE), Math.random() * 10);
    return new Brain(rand_hex(BRAIN_SIZE), Math.random() * 5)
  }

  getMove() {
    const move = parseInt(this.program[Math.floor(this.clock)], 16);
    
    const angle_move = ((move+1) % 3) - 1;
    const linear_move = (((move >>> 2)+1) % 3) - 1;

  
    this.clock = (this.clock + this.clock_rate) % this.program.length;

    return [angle_move, linear_move];
  }


  mut_copy() {

    let new_prog = '';
    for (let i = 0; i < this.program.length; i++) {
      if (Math.random() < (1/this.program.length)) {
        new_prog += rand_hex(1);
      }
      else {
        new_prog += this.program[i];
      }
    }

    return new Brain(new_prog, this.clock_rate * (1 + rand_in_range(-CLOCK_RATE_MUT_SIZE, CLOCK_RATE_MUT_SIZE)));

  }


}



class Gene {
  constructor(size, color, split_ratio, split_thresh, brain) {
    this.size = size;
    this.color = color;
    this.split_ratio = split_ratio;
    this.split_thresh = split_thresh;
    this.brain = brain;
    this.brain.setSize(size);
  }

  static rand() {
    return new Gene(
                    rand_in_range(MIN_SIZE, MAX_SIZE), 
                    rand_color(), 
                    Math.random(),
                    Math.random() * 5,
                    NeuralBrain.rand());
  }

  getSize() {
    return this.size;
  }

  getColor() {
    return this.color;
  }

  getSplitRatio() {
    return this.split_ratio;
  }

  getSplitThresh() {
    return this.split_thresh;
  }

  mut_copy() {
    if (Math.random() < MUT_PROB) {
      const new_size = clip(this.size * (1 + rand_in_range(-SIZE_MUT_FRAC, SIZE_MUT_FRAC)), 
                            MIN_SIZE, 
                            MAX_SIZE);
      const color = Gene.mut_color(this.color);  
      const split_ratio = Gene.mut_split_ratio(this.split_ratio);
      const split_thresh = Gene.mut_split_thresh(this.split_thresh);
      const brain = this.brain.mut_copy();
      return new Gene(new_size, 
                      color,
                      split_ratio,
                      split_thresh,
                      brain);
    }
    else {
      return new Gene(
                      this.size, 
                      this.color, 
                      this.split_ratio, 
                      this.split_thresh,
                      this.brain);
    }
  }

  static mut_color(color) {

    const r = clip((parseInt(color.slice(0,2), 16) + 
                Math.floor(Math.random() * COLOR_MUT_SIZE) - Math.floor( 
                COLOR_MUT_SIZE/2)), 0, 255).toString(16);
    const g = clip((parseInt(color.slice(2,4), 16) + 
                Math.floor(Math.random() * COLOR_MUT_SIZE) - Math.floor( 
                COLOR_MUT_SIZE/2)), 0, 255).toString(16);
    const b = clip((parseInt(color.slice(4,6), 16) + 
                Math.floor(Math.random() * COLOR_MUT_SIZE) - Math.floor( 
                COLOR_MUT_SIZE/2)), 0, 255).toString(16);

    const new_color =  ("0" + r).slice(-2) + ("0" + g).slice(-2) + ("0" + b).slice(-2); 

    return new_color;
  }

  static mut_split_ratio(sr) {
    return clip(sr + rand_in_range(-1,1) * SPLIT_R_MUT_SIZE, 0, 1);
  }

  static mut_split_thresh(st) {
    return clip(st + rand_in_range(-1,1) * SPLIT_T_MUT_SIZE, MIN_SPLIT_THRESH, MAX_SPLIT_THRESH);
  }
}



class Pop {
  constructor(pos, vel, energy, angle,  gene) {
    this.gene = gene;
    this.size = gene.getSize();
    this.size_sq = this.size * this.size;
    this.color = this.gene.getColor();
    this.pointer_color = invert_color(this.color)

    this.pos = pos;
    this.vel = vel;
    this.angle_vec = Vec2.from_angle(angle);
    this.energy = energy;
    this.alive = true;
  }

  static rand() {
    return new Pop(
                    Vec2.rand(), 
                    Vec2.rand().mulSc(2).addSc(-1).mulSc(MAX_INIT_VEL),
                    1,
                    rand_angle(),
                    Gene.rand()
                  );
  }

  update() {
    this.pos = this.pos.addVec(this.vel);
    const vec_to_add = new Vec2(-Math.floor(this.pos.getX()), 
                             -Math.floor(this.pos.getY()));
    this.pos = this.pos.addVec(vec_to_add);
    this.vel = this.vel.mulSc(FRICTION_COEFF);
    
    this.energy -= this.size_sq * ENERGY_SIZE_COST;
    this.energy += BASE_ENERGY_INCREASE;
    if (this.energy < 0) {
      this.alive = false;
    }

    this.gene.brain.setAngleVec(this.angle_vec);
    this.gene.brain.setEnergy(this.energy);
    const [ang_move, lin_move] = this.gene.brain.getMove();
    if (!ang_move || !lin_move) {
      this.alive = false;
    }

    if (this.angle_vec != 0) {
      this.angle_vec = this.angle_vec.rotate(ang_move*ANG_MOVE_SPEED);
    }
    this.vel = this.vel.addVec(this.angle_vec.mulSc(lin_move * LIN_MOVE_SPEED));
    
    this.energy -= ENERGY_MOVE_COST * Math.abs(lin_move);

    
  }

  collidesWith(other, dist) {
    return (dist < Math.max(Math.min(this.size, other.getSize()), (this.size + other.getSize())/4));
  }

  interact(other) {
    const oPos = other.getPos().mulSc(-1);
    const diff_vec = this.pos.addVec(oPos);
    const dist = diff_vec.norm();

    if (this.collidesWith(other, dist)) {
      if (this.size_sq < other.getSizeSq() * EAT_SIZE_THRESH) {
        other.eat(this);
      }
      else if (other.getSizeSq() < this.size_sq * EAT_SIZE_THRESH) {
        this.eat(other);
      }
      else {
        other.enervate();
        this.enervate();
      }
    }

    this.gene.brain.sense(dist, diff_vec, other.getSize());
    other.gene.brain.sense(dist, diff_vec, this.size);
  }

  isAlive() {
    return this.alive;
  }

  kill() {
    this.energy = 0;
    this.alive = false;
  }

  eat(other) {
    this.energy += other.getEnergy();
    other.kill();
  }

  getEnergy() {
    return this.energy;
  }

  enervate() {
    this.energy -= ENERGY_OVERLAP_COST;
  }

  makeChild() {
    const new_pos = this.pos.addVec(
      Vec2.rand_unit().mulSc(2 * this.size)
    );

    const  dVel = Vec2.rand().mulSc(2).addSc(-1).mulSc(VEL_JITTER);
    const new_vel = this.vel.addVec(
      dVel
    );

    const energyToGive = this.energy * this.gene.getSplitRatio();
    this.energy = this.energy - energyToGive;

    return new Pop(new_pos, new_vel, energyToGive, rand_angle(), this.gene.mut_copy());
  }

  canMakeChild() {
    return (this.energy > this.gene.getSplitThresh());
  }

  getX() {
    return this.pos.getX();
  }

  getY() {
    return this.pos.getY();
  }

  getPos() {
    return this.pos;
  }

  getVel() {
    return this.vel;
  }

  getSize() {
    return this.size;
  }

  getSizeSq() {
    return this.size_sq;
  }

  getAngleVec() {
    return this.angle_vec;
  }

  getColor() {
    return this.color;
  }

  getPointerColor() {
    return this.pointer_color;
  }
}



class EvoSim {
  constructor() {
    this.pop = [];
    this.step = this.step.bind(this);
  }

  step() {
    while (this.pop.length < MIN_POP) {
      this.pop.push(Pop.rand());
    }

    for (let i=0; i<this.pop.length; i++) {
      for (let j=i+1; j<this.pop.length; j++) {
        
      
        this.pop[i].interact(this.pop[j]);

        if (!this.pop[i].isAlive()) {
          this.pop.splice(i,1);
          i--;
          break;
        }
        if (!this.pop[j].isAlive()) {
          this.pop.splice(j,1);
          j--;
        }
      }
    }

    for (let k=0; k < this.pop.length; k++) {
      if (this.pop[k].canMakeChild()) {
        this.pop.push(this.pop[k].makeChild());
      }

      this.pop[k].update();
      
      if (!this.pop[k].isAlive()) {
        this.pop.splice(k,1);
        k--;
      }

      if (this.pop.length > MAX_POP) {
        this.pop = this.pop.slice(0, MAX_POP);
      }
    }
  }

  getPop() {
    return this.pop;
  }
}


class CanvasDrawer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.scale = Math.min(canvas.height, canvas.width)/2;
    this.origin = new Vec2(0, 0);
    this.adj = new Vec2(canvas.width/2, canvas.height/2);
  }

  transform(v) {
    // (v * 2 - 1) * scale + origin
    return v.mulSc(2).addSc(-1).mulSc(this.scale).addVec(this.origin).addVec(this.adj);
  }

  drawPop(pop) {
    const canvasPt = this.transform(pop.getPos());
    const ptSize = pop.getSize()*this.scale;
    const outsidePt = canvasPt.addVec(pop.getAngleVec().mulSc(ptSize*1.1));

    this.ctx.fillStyle = "#" + pop.getColor();
    this.ctx.beginPath();
    this.ctx.arc(canvasPt.getX(), canvasPt.getY(), ptSize, 0, 2*Math.PI);
    this.ctx.fill();
    
    this.ctx.strokeStyle = "#" + pop.getPointerColor();
    this.ctx.beginPath()
    this.ctx.moveTo(canvasPt.getX(), canvasPt.getY());
    this.ctx.lineTo(outsidePt.getX(), outsidePt.getY());
    this.ctx.stroke();
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  zoom(scrollDir, x, y) {
    const coordPre = new Vec2(x,y);

    const unitCoord = coordPre.addVec(this.origin.addVec(this.adj).mulSc(-1)).mulSc(1/this.scale);

    const dscale = 1+0.1*Math.sign(-scrollDir)
    this.scale = Math.min(Math.max(dscale*this.scale, 1), this.canvas.height * 100);
    
    const coordPost = unitCoord.mulSc(this.scale).addVec(this.origin).addVec(this.adj);
    const coordDiff = coordPost.mulSc(-1).addVec(coordPre);
    
    // Set the origin so that the position in the sim coordinates of the
    // mouse does not change after the scale is changed
    this.origin = this.origin.addVec(coordDiff.mulSc(1));
  }

  move(dx,dy) {
    this.origin = this.origin.addVec(new Vec2(dx,dy));
  }
}


const main = () => {
  const canvas = document.getElementById('mainCanvas');
  console.log(canvas);
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  window.addEventListener('resize', (event)=>{
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }, true);


  const drawer = new CanvasDrawer(canvas);
  window.addEventListener('wheel', (e)=>{
    drawer.zoom(e.deltaY, e.x, e.y);
  });

  
  let dragging = false;
  window.addEventListener('mousedown', (e)=>{
    if (e && (e.which == 2 || e.button == 4)) {
      dragging = true;
    }
  });

  window.addEventListener('mousemove', (e)=>{
    if (dragging) {
      drawer.move(e.movementX, e.movementY);
    }
  });

  window.addEventListener('mouseup', (e)=>{
    if (e && (e.which == 2 || e.button == 4)) {
      dragging = false;
    }
  });


  const sim = new EvoSim();

  const mainLoop = () => {

    sim.step();
   
    drawer.clear();
    sim.getPop().forEach((pop) => {
      drawer.drawPop(pop);
    });

    requestAnimationFrame(mainLoop);
  };

  mainLoop();

};

window.onload = main;




