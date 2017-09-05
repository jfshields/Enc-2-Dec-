class RRN_2_MLP_NN():
    def __init__(self, t_in, t_out, d_in, d_out, n_units, starter_learning_rate, reg_val, alt= 'N'):
        self.self= self
        self.t_in= t_in
        self.d_in = d_in
        self.t_out= t_out
        self.d_out = d_out
        self.n_units= n_units
        self.starter_learning_rate= starter_learning_rate
        self.reg_val= reg_val        
        self.atl= alt
        self.n_epoch= 0
        self.d_loss= {}
        self.m_name= str(alt)+ '_'+ str(t_in)+ '_'+ str(t_out)+ '_'+ str(d_in)+ '_'+ str(d_out)+ '_'+ str(n_units) + '_' + str(reg_val)
        self.initilise_model()
        if self.atl== 'Y':
            self.initilise_model2()

    def initilise_model(self):
        tf.reset_default_graph()

        self.x_i = tf.placeholder("float", [None, self.t_in, self.d_in])
        self.y_i = tf.placeholder("float", [None, self.t_out, self.d_out])
        self.t_i = tf.placeholder("float", [None, (self.t_in+ self.t_out), 1])

        self.cell = tf.contrib.rnn.BasicRNNCell(self.n_units)
        self.outputs = tf.nn.dynamic_rnn(self.cell, self.x_i, dtype= tf.float32)

        self.n_l0= 100
        self.n_l1= 100
        self.w0 = tf.Variable(tf.random_normal([self.n_units, self.n_l0], stddev=0.35), name="w0")
        self.b0 = tf.Variable(tf.random_normal([self.n_l0], stddev=0.35), name="b0")
        self.w1 = tf.Variable(tf.random_normal([self.n_l0, self.n_l1], stddev=0.35), name="w0")
        self.b1 = tf.Variable(tf.random_normal([self.n_l1], stddev=0.35), name="b0")
        self.w2 = tf.Variable(tf.random_normal([self.n_l1, self.t_out], stddev=0.35), name="w0")
        self.b2 = tf.Variable(tf.random_normal([self.t_out], stddev=0.35), name="b0")

        self.outputs_a0 = tf.add(tf.matmul(self.outputs[-1], self.w0), self.b0)
        self.outputs_z0 = tf.nn.sigmoid(self.outputs_a0)
        self.outputs_a1 = tf.add(tf.matmul(self.outputs_a0, self.w1), self.b1)
        self.outputs_z1 = tf.nn.sigmoid(self.outputs_a1)
        self.outputs_a2 = tf.add(tf.matmul(self.outputs_z1, self.w2), self.b2)
        self.outputs_z2 = tf.nn.relu6(self.outputs_a2)        
        
        self.y_hat = tf.reshape(self.outputs_z2, [-1, self.t_out, self.d_out])
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y_i, self.y_hat))
        
        self.l2 = float(self.reg_val) * sum(
                tf.nn.l2_loss(tf_var)
                    for tf_var in tf.trainable_variables()
                    if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        self.lossL2 = self.loss+ self.l2        
                
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 100000, 0.96, staircase=True)
        self.optimiser = tf.train.AdamOptimizer(self.learning_rate).minimize(self.lossL2)
        
    def initilise_model2(self):
        
        self.n_l0= 200
        self.n_l1= 200
        self.w0 = tf.Variable(tf.random_normal([(self.n_units* self.t_in), self.n_l0], stddev=0.35), name="w0")
        self.b0 = tf.Variable(tf.random_normal([self.n_l0], stddev=0.35), name="b0")
        self.w1 = tf.Variable(tf.random_normal([self.n_l0, self.n_l1], stddev=0.35), name="w0")
        self.b1 = tf.Variable(tf.random_normal([self.n_l1], stddev=0.35), name="b0")
        self.w2 = tf.Variable(tf.random_normal([self.n_l1, self.t_out], stddev=0.35), name="w0")
        self.b2 = tf.Variable(tf.random_normal([self.t_out], stddev=0.35), name="b0")

        self.outputs_r= tf.reshape(self.outputs[0], [-1, (self.n_units* self.t_in)])
        
        self.outputs_a0 = tf.add(tf.matmul(self.outputs_r, self.w0), self.b0)
        self.outputs_z0 = tf.nn.sigmoid(self.outputs_a0)
        self.outputs_a1 = tf.add(tf.matmul(self.outputs_a0, self.w1), self.b1)
        self.outputs_z1 = tf.nn.sigmoid(self.outputs_a1)
        self.outputs_a2 = tf.add(tf.matmul(self.outputs_z1, self.w2), self.b2)
        self.outputs_z2 = tf.nn.relu6(self.outputs_a2)  
        
        self.y_hat = tf.reshape(self.outputs_z2, [-1, self.t_out, self.d_out])
        
    
    def train_full(self, df_x2, df_y2, epoch, batch_sz):
        self.iter_ = int(df_x2.shape[0] / batch_sz)
        print('Iters: %s. Epoch:  %s' % (self.iter_, epoch))
        for e in range(epoch):
            e_loss = 0
            for i in range(self.iter_):
                iter_dict = {self.x_i: df_x2[(i * batch_sz):((i + 1) * batch_sz)],
                             self.y_i: df_y2[(i * batch_sz):((i + 1) * batch_sz)]}
                sess.run(self.optimiser, feed_dict=iter_dict)
                i_loss= sess.run(self.loss, feed_dict=iter_dict)
                e_loss+= i_loss
            print('Epoch %s, loss %s' % (e, e_loss))
        return e_loss
    
    def tr_te_full(self, df_x2, df_y2, epoch, batch_sz, df_xtest, df_ytest):
        self.iter_ = int(df_x2.shape[0] / batch_sz)
        print('Iters: %s. Epoch:  %s' % (self.iter_, epoch))
        for e in range(epoch):
            e_loss = 0
            for i in range(self.iter_):
                #self.rand_input(p= 0.9)
                iter_dict = {self.x_i: df_x2[(i * batch_sz):((i + 1) * batch_sz)],
                             self.y_i: df_y2[(i * batch_sz):((i + 1) * batch_sz)]}
                sess.run(self.optimiser, feed_dict= iter_dict)
                i_loss= sess.run(self.loss, feed_dict= iter_dict)
                e_loss+= i_loss
                #self.rand_reset()
                
                if i== round((self.iter_/ 2), 0):
                    trai_dict = {self.x_i: df_x2,
                         self.y_i: df_y2} 
                    tr_loss= sess.run(self.loss, feed_dict= trai_dict)
                    test_dict = {self.x_i: df_xtest,
                         self.y_i: df_ytest} 
                    te_loss= sess.run(self.loss, feed_dict= test_dict)
                    print('Epoch %s.5, train loss %s, test loss %s' % (self.n_epoch, tr_loss, te_loss))
                if (i== round((self.iter_/ 2), 0)) & (self.n_epoch== 1):
                    break
            
            self.n_epoch+= 1
            trai_dict = {self.x_i: df_x2,
                         self.y_i: df_y2} 
            tr_loss= sess.run(self.loss, feed_dict= trai_dict)
            test_dict = {self.x_i: df_xtest,
                         self.y_i: df_ytest} 
            te_loss= sess.run(self.loss, feed_dict= test_dict)
            print('Epoch %s, train loss %s, test loss %s' % (self.n_epoch, tr_loss, te_loss))
        return e_loss
    
    def save_model(self, sess, NN_name= 'default'):
        file_path= './'+ NN_name + '/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        saver = tf.train.Saver()
        saver.save(sess, file_path+ 'model.checkpoint')
        print('Model saved')

    def load_model(self, sess, NN_name= 'default'):
        file_path = './'+ NN_name + '/'
        saver = tf.train.Saver()
        saver.restore(sess, file_path+ 'model.checkpoint')
        print('Model loaded')



class Enc_2_Dec_NN():
    def __init__(self, t_in, t_out, d_in, d_out, n_units_e, n_stacks_e, n_units_d, starter_learning_rate, reg_val):
        self.self= self
        self.t_in= t_in
        self.t_out= t_out
        self.d_in = d_in
        self.d_out = d_out
        self.n_units_e= n_units_e
        self.n_stacks_e= n_stacks_e
        self.n_units_d= n_units_d
        self.starter_learning_rate= starter_learning_rate
        self.reg_val= reg_val 
        self.n_epoch= 0
        self.d_loss= {}
        self.m_name= 'M_'+ str(t_in)+ '_'+ str(t_out)+ '_'+ str(d_in)+ '_'+ str(d_out)+ '_'+ str(n_units_e)+ '_'+ str(n_stacks_e)+ '_'+ str(n_units_d)
        self.weights= {}
        self.calcs = {}
        self.initilise_model()  
    
    def initilise_model(self):
        tf.reset_default_graph()
        self.x_i = tf.placeholder("float", [None, self.t_in, self.d_in])
        self.y_i = tf.placeholder("float", [None, self.t_out, self.d_out])
        self.t_i = tf.placeholder("float", [None, (self.t_in+ self.t_out), 1])
        self.rand= tf.placeholder("float", [None, self.t_out, 1])

        self.stack_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.n_units_e) for x in range(self.n_stacks_e)], state_is_tuple= True)
        self.outputs, self.states = tf.nn.dynamic_rnn(self.stack_cell, self.x_i, dtype= tf.float32)
        
        self.w1= tf.Variable(tf.random_uniform([self.n_units_e, self.n_units_e], minval=-0.08, maxval= 0.08), name="weights") 
        self.b1= tf.Variable(tf.random_uniform([self.n_units_e], minval=-0.08, maxval= 0.08), name="weights") 
       
        self.w4= tf.Variable(tf.random_uniform([(self.t_in- 1)* self.n_units_e, (self.n_units_e)], minval=-0.08, maxval= 0.08), name="weights") 
        self.dinputall= tf.matmul(tf.reshape(self.outputs[:, :-1], [-1, (self.t_in- 1)* self.n_units_e]), self.w4)
        
        self.h_prev0= self.states[-1].h
        self.c_prev0= self.states[-1].c
        self.h_prev1= self.states[-2 :-1][0].h
        self.c_prev1= self.states[-2 :-1][0].c
        self.h_prev2= self.states[-3 :-2][0].h
        self.c_prev2= self.states[-3 :-2][0].c
        
        self.x_prev0= tf.matmul(self.outputs[:, -1], self.w1) 
        
        self.cell_weig(0, 0)
        self.cell_calc(0, 0, self.x_prev0, self.h_prev0, self.c_prev0)
        self.output0= tf.expand_dims(self.calcs[(0, 0)]['o_1'], axis= 1)
        #Row 1, col 0 
        self.cell_weig(1, 0)
        self.cell_calc(1, 0, self.calcs[(0, 0)]['o_1'], self.h_prev1, self.c_prev1)
        self.output1= tf.expand_dims(self.calcs[(1, 0)]['o_1'], axis= 1)
        #Row 2, col 0 
        self.cell_weig(2, 0)
        self.cell_calc(2, 0, self.calcs[(1, 0)]['o_1'], self.h_prev2, self.c_prev2)
        self.output2= tf.expand_dims(self.calcs[(2, 0)]['o_1'], axis= 1)
        for i in range(1, 12): #self.n_steps_d
            #Row 0, col i
            self.cell_weig(0, i)
            self.cell_calc(0, i, self.calcs[(2, (i- 1))]['o_1'], self.calcs[(0, (i- 1))]['h_1'], self.calcs[(0, (i- 1))]['c_1'])
            self.output0= tf.concat([self.output0, tf.expand_dims(self.calcs[(0, i)]['o_1'], axis= 1)], axis= 1)
            #Row 1, col i
            self.cell_weig(1, i)
            self.cell_calc(1, i, self.calcs[(0, i)]['o_1'], self.calcs[(1, (i- 1))]['h_1'], self.calcs[(1, (i- 1))]['c_1'])
            self.output1= tf.concat([self.output1, tf.expand_dims(self.calcs[(1, i)]['o_1'], axis= 1)], axis= 1)
            #Row 2, col i
            self.cell_weig(2, i)
            self.cell_calc(2, i, self.calcs[(1, i)]['o_1'], self.calcs[(2, (i- 1))]['h_1'], self.calcs[(2, (i- 1))]['c_1'])
            self.output2= tf.concat([self.output2, tf.expand_dims(self.calcs[(2, i)]['o_1'], axis= 1)], axis= 1)  

        
        self.w3= tf.Variable(tf.random_uniform([self.n_units_d, self.t_out, self.d_out], minval=-0.08, maxval= 0.08), name="weights")

        self.output3= tf.matmul(self.output2[:, 0], self.w3[:, 0])
        for i in range(1, 12):
            self.output3= tf.concat([self.output3, tf.matmul(self.output2[:, i], self.w3[:, i])], 1)
            
        self.y_hat= tf.reshape(self.output3, [-1, self.t_out, self.d_out])
        
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y_i, self.y_hat))
        #tf.losses.mean_squared_error
        #tf.losses.absolute_difference
        
        self.l2 = float(self.reg_val) * sum(
                tf.nn.l2_loss(tf_var)
                    for tf_var in tf.trainable_variables()
                    if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        self.lossL2 = self.loss+ self.l2        
        
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 100000, 0.96, staircase=True)
        self.optimiser = tf.train.AdamOptimizer(self.learning_rate).minimize(self.lossL2)
        #AdamOptimizer
        #RMSPropOptimizer
        #GradientDescentOptimizer

    def cell_weig(self, i, j):
        self.weights[(i, j)] = {}
        self.weights[(i, j)]['W_f'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['U_f'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['b_f'] = tf.Variable(tf.random_uniform([self.n_units_d], minval=0, maxval= 1), name="weights")
        self.weights[(i, j)]['W_i'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['U_i'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['b_i'] = tf.Variable(tf.random_uniform([self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['W_o'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['U_o'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['b_o'] = tf.Variable(tf.random_uniform([self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['W_c'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['U_c'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['b_c'] = tf.Variable(tf.random_uniform([self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
    
    def cell_calc(self, i, j, x_this, h_prev, c_prev):
        self.calcs[(i, j)] = {}
        self.calcs[(i, j)]['f_1'] = tf.sigmoid(tf.add((tf.matmul(x_this, self.weights[(i, j)]['W_f'])+ tf.matmul(h_prev, self.weights[(i, j)]['U_f'])), self.weights[(i, j)]['b_f']))
        self.calcs[(i, j)]['i_1'] = tf.sigmoid(tf.add((tf.matmul(x_this, self.weights[(i, j)]['W_i']) + tf.matmul(h_prev, self.weights[(i, j)]['U_i'])), self.weights[(i, j)]['b_i']))
        self.calcs[(i, j)]['o_1'] = tf.nn.relu(tf.add((tf.matmul(x_this, self.weights[(i, j)]['W_o']) + tf.matmul(h_prev, self.weights[(i, j)]['U_o'])), self.weights[(i, j)]['b_o']))
        self.calcs[(i, j)]['c_1'] = tf.multiply(self.calcs[(i, j)]['f_1'], c_prev) + tf.multiply(self.calcs[(i, j)]['i_1'], tf.tanh( tf.matmul(h_prev, self.weights[(i, j)]['W_c']) + tf.matmul(h_prev, tf.add(self.weights[(i, j)]['U_c'], self.weights[(i, j)]['b_c']))))
        self.calcs[(i, j)]['h_1'] = tf.multiply(self.calcs[(i, j)]['o_1'], tf.tanh(self.calcs[(i, j)]['c_1']))  
        
        
    def train_full(self, df_x2, df_y2, epoch, batch_sz):
        self.iter_ = int(df_x2.shape[0] / batch_sz)
        print('Iters: %s. Epoch:  %s' % (self.iter_, epoch))
        for e in range(epoch):
            e_loss = 0
            for i in range(self.iter_):
                iter_dict = {self.x_i: df_x2[(i * batch_sz):((i + 1) * batch_sz)],
                             self.y_i: df_y2[(i * batch_sz):((i + 1) * batch_sz)]}
                sess.run(self.optimiser, feed_dict=iter_dict)
                i_loss= sess.run(self.loss, feed_dict=iter_dict)
                e_loss+= i_loss
            print('Epoch %s, loss %s' % (e, e_loss))
        return e_loss
    
    def tr_te_full(self, df_x2, df_y2, epoch, batch_sz, df_xtest, df_ytest):
        self.iter_ = int(df_x2.shape[0] / batch_sz)
        print('Iters: %s. Epoch:  %s' % (self.iter_, epoch))
        for e in range(epoch):
            e_loss = 0
            for i in range(self.iter_):
                iter_dict = {self.x_i: df_x2[(i * batch_sz):((i + 1) * batch_sz)],
                             self.y_i: df_y2[(i * batch_sz):((i + 1) * batch_sz)]}
                sess.run(self.optimiser, feed_dict= iter_dict)
                i_loss= sess.run(self.loss, feed_dict= iter_dict)
                e_loss+= i_loss
                
                if i== round((self.iter_/ 2),0):
                    trai_dict = {self.x_i: df_x2,
                         self.y_i: df_y2} 
                    tr_loss= sess.run(self.loss, feed_dict= trai_dict)
                    test_dict = {self.x_i: df_xtest,
                         self.y_i: df_ytest} 
                    te_loss= sess.run(self.loss, feed_dict= test_dict)
                    print('Epoch %s.5, train loss %s, test loss %s' % (self.n_epoch, tr_loss, te_loss))
            
            self.n_epoch+= 1
            trai_dict = {self.x_i: df_x2,
                         self.y_i: df_y2} 
            tr_loss= sess.run(self.loss, feed_dict= trai_dict)
            test_dict = {self.x_i: df_xtest,
                         self.y_i: df_ytest} 
            te_loss= sess.run(self.loss, feed_dict= test_dict)
            print('Epoch %s, train loss %s, test loss %s' % (self.n_epoch, tr_loss, te_loss))
        return e_loss
    
    def save_model(self, sess, NN_name= 'default'):
        file_path= './'+ NN_name + '/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        saver = tf.train.Saver()
        saver.save(sess, file_path+ 'model.checkpoint')
        print('Model saved')

    def load_model(self, sess, NN_name= 'default'):
        file_path = './' + NN_name + '/'
        saver = tf.train.Saver()
        saver.restore(sess, file_path+ 'model.checkpoint')
        print('Model loaded')


# Seq_alt2Seq
class Enc_2_DecALT():
    def __init__(self, t_in, t_out, d_in, d_out, n_units_e, n_stacks_e, n_units_d, starter_learning_rate, reg_val):
        self.self= self
        self.t_in= t_in
        self.t_out= t_out
        self.d_in = d_in
        self.d_out = d_out
        self.n_units_e= n_units_e
        self.n_stacks_e= n_stacks_e
        self.n_units_d= n_units_d
        self.starter_learning_rate= starter_learning_rate
        self.reg_val= reg_val 
        self.n_epoch= 0
        self.d_loss= {}
        self.m_name= 'M_Alt_'+ str(t_in)+ '_'+ str(t_out)+ '_'+ str(d_in)+ '_'+ str(d_out)+ '_'+ str(n_units_e)+ '_'+ str(n_stacks_e)+ '_'+ str(n_units_d)
        self.weights= {}
        self.calcs = {}
        self.initilise_model()  
    
    def initilise_model(self):
        tf.reset_default_graph()
        self.x_i = tf.placeholder("float", [None, self.t_in, self.d_in])
        self.y_i = tf.placeholder("float", [None, self.t_out, self.d_out])
        self.t_i = tf.placeholder("float", [None, (self.t_in+ self.t_out), 1])

        self.stack_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.n_units_e) for x in range(self.n_stacks_e)], state_is_tuple= True)
        self.outputs, self.states = tf.nn.dynamic_rnn(self.stack_cell, self.x_i, dtype= tf.float32)
        
        self.w1= tf.Variable(tf.random_uniform([self.n_units_e, self.n_units_e], minval=-0.08, maxval= 0.08), name="weights") 
        self.b1= tf.Variable(tf.random_uniform([self.n_units_e], minval=-0.08, maxval= 0.08), name="weights") 
        self.w2= tf.Variable(tf.random_uniform([self.d_out, self.n_units_e], minval=-0.08, maxval= 0.08), name="weights") 
        self.b2= tf.Variable(tf.random_uniform([self.n_units_e], minval=-0.08, maxval= 0.08), name="weights") 
        
        self.w4= tf.Variable(tf.random_uniform([(self.t_in- 1)* self.n_units_e, (self.n_units_e)], minval=-0.08, maxval= 0.08), name="weights") 
        self.dinputall= tf.matmul(tf.reshape(self.outputs[:, :-1], [-1, (self.t_in- 1)* self.n_units_e]), self.w4)
        
        self.x_prev0= self.outputs[:, -1]
        self.h_prev0= self.states[-1].h
        self.c_prev0= self.states[-1].c
        self.h_prev1= self.states[-2 :-1][0].h
        self.c_prev1= self.states[-2 :-1][0].c
        self.h_prev2= self.states[-3 :-2][0].h
        self.c_prev2= self.states[-3 :-2][0].c
 
        self.w_at= tf.Variable(tf.random_uniform([(self.t_in* self.n_units_e), (self.t_out* self.n_units_d)], minval=-0.08, maxval= 0.08), name="weights")
        self.x_att= tf.reshape(tf.matmul(tf.reshape(tf.sigmoid(self.outputs[:, :]), [-1, (24* self.n_units_e)]), self.w_at), [-1, self.t_out, self.n_units_d])

        self.y_trans= tf.expand_dims(tf.matmul(self.y_i[:, 0], self.w2), 1)
        for i in range(1, 12):
            self.y_trans= tf.concat([self.y_trans, tf.expand_dims(tf.matmul(self.y_i[:, i], self.w2), 1)], 1)   
        
        self.cell_weigALT(0, 0)
        self.cell_calcALT(0, 0, tf.add(self.x_prev0, self.x_att[:, 0]), self.h_prev0, self.c_prev0)    
        
        self.output0= tf.expand_dims(self.calcs[(0, 0)]['o_1'], axis= 1)
        #Row 1, col 0 
        self.cell_weigALT(1, 0)
        self.cell_calcALT(1, 0, self.calcs[(0, 0)]['o_1'], self.h_prev1, self.c_prev1)
        self.output1= tf.expand_dims(self.calcs[(1, 0)]['o_1'], axis= 1)
        #Row 2, col 0 
        self.cell_weigALT(2, 0)
        self.cell_calcALT(2, 0, self.calcs[(1, 0)]['o_1'], self.h_prev2, self.c_prev2)
        self.output2= tf.expand_dims(self.calcs[(2, 0)]['o_1'], axis= 1)
        for i in range(1, 12): #self.n_steps_d
            #Row 0, col i
            self.cell_weigALT(0, i)
            self.cell_calcALT(0, i, tf.add(self.calcs[(2, (i- 1))]['o_1'], self.x_att[:, i]), self.calcs[(0, (i- 1))]['h_1'], self.calcs[(0, (i- 1))]['c_1'])
            self.output0= tf.concat([self.output0, tf.expand_dims(self.calcs[(0, i)]['o_1'], axis= 1)], axis= 1)
            #Row 1, col i
            self.cell_weigALT(1, i)
            self.cell_calcALT(1, i, self.calcs[(0, i)]['o_1'], self.calcs[(1, (i- 1))]['h_1'], self.calcs[(1, (i- 1))]['c_1'])
            self.output1= tf.concat([self.output1, tf.expand_dims(self.calcs[(1, i)]['o_1'], axis= 1)], axis= 1)
            #Row 2, col i
            self.cell_weigALT(2, i)
            self.cell_calcALT(2, i, self.calcs[(1, i)]['o_1'], self.calcs[(2, (i- 1))]['h_1'], self.calcs[(2, (i- 1))]['c_1'])
            self.output2= tf.concat([self.output2, tf.expand_dims(self.calcs[(2, i)]['o_1'], axis= 1)], axis= 1)  

        
        self.w3= tf.Variable(tf.random_uniform([self.n_units_d, self.t_out, self.d_out], minval=-0.08, maxval= 0.08), name="weights")

        self.output3= tf.matmul(self.output2[:, 0], self.w3[:, 0])
        for i in range(1, 12):
            self.output3= tf.concat([self.output3, tf.matmul(self.output2[:, i], self.w3[:, i])], 1)
            
        self.y_hat= tf.nn.relu(tf.reshape(self.output3, [-1, self.t_out, self.d_out]))
        
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y_i, self.y_hat))
        
        self.l2 = float(self.reg_val) * sum(
                tf.nn.l2_loss(tf_var)
                    for tf_var in tf.trainable_variables()
                    if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        self.lossL2 = self.loss+ self.l2        
        
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 100000, 0.96, staircase=True)
        self.optimiser = tf.train.AdamOptimizer(self.starter_learning_rate).minimize(self.lossL2)

    def cell_weigALT(self, i, j):
        self.weights[(i, j)] = {}
        self.weights[(i, j)]['W_f'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['U_f'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['C_f'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['b_f'] = tf.Variable(tf.random_uniform([self.n_units_d], minval=0, maxval= 1), name="weights")
        
        self.weights[(i, j)]['W_i'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['U_i'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['C_i'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['b_i'] = tf.Variable(tf.random_uniform([self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")

        self.weights[(i, j)]['W_o'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['U_o'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['C_o'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['b_o'] = tf.Variable(tf.random_uniform([self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")

        self.weights[(i, j)]['W_c'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['U_c'] = tf.Variable(tf.random_uniform([self.n_units_d, self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
        self.weights[(i, j)]['b_c'] = tf.Variable(tf.random_uniform([self.n_units_d], minval=-0.08, maxval= 0.08), name="weights")
    
    
    def cell_calcALT(self, i, j, x_this, h_prev, c_prev):
        self.calcs[(i, j)] = {}
        self.calcs[(i, j)]['f_1'] = tf.sigmoid(tf.add((tf.matmul(x_this, self.weights[(i, j)]['W_f'])+ tf.matmul(h_prev, self.weights[(i, j)]['U_f'])+ tf.matmul(c_prev, self.weights[(i, j)]['C_f'])), self.weights[(i, j)]['b_f']))
        self.calcs[(i, j)]['i_1'] = tf.sigmoid(tf.add((tf.matmul(x_this, self.weights[(i, j)]['W_i'])+ tf.matmul(h_prev, self.weights[(i, j)]['U_i'])+ tf.matmul(c_prev, self.weights[(i, j)]['C_i'])), self.weights[(i, j)]['b_i']))
        self.calcs[(i, j)]['o_1'] = tf.nn.relu(tf.add((tf.matmul(x_this, self.weights[(i, j)]['W_o'])+ tf.matmul(h_prev, self.weights[(i, j)]['U_o'])+ tf.matmul(c_prev, self.weights[(i, j)]['C_o'])), self.weights[(i, j)]['b_o']))
        self.calcs[(i, j)]['c_1'] = tf.multiply(self.calcs[(i, j)]['f_1'], c_prev) + tf.multiply(self.calcs[(i, j)]['i_1'], tf.tanh( tf.matmul(h_prev, self.weights[(i, j)]['W_c']) + tf.matmul(h_prev, tf.add(self.weights[(i, j)]['U_c'], self.weights[(i, j)]['b_c']))))
        self.calcs[(i, j)]['h_1'] = tf.multiply(self.calcs[(i, j)]['o_1'], tf.tanh(self.calcs[(i, j)]['c_1']))  
        
    def rand_input(self, p):
        for i in range(1, 12):
            if random.uniform(0, 1)> p:
                #print('Change', i)
                self.cell_calcALT(0, i, self.calcs[(2, (i- 1))]['o_1'], self.calcs[(0, (i- 1))]['h_1'], self.calcs[(0, (i- 1))]['c_1'])
            else:
                pass
            
    def rand_reset(self):
        for i in range(1, 12):
            #print('Change back', i)
            self.cell_calcALT(0, i, self.y_trans[:, (i- 1)], self.calcs[(0, (i- 1))]['h_1'], self.calcs[(0, (i- 1))]['c_1'])

    def chang_input(self):
        self.cell_calcALT(0, 4, self.y_trans[3], self.calcs[(0, (4- 1))]['h_1'], self.calcs[(0, (4- 1))]['c_1'])

    def chang_reset(self):
        self.cell_calcALT(0, 4, self.calcs[(2, (4- 1))]['o_1'], self.calcs[(0, (4- 1))]['h_1'], self.calcs[(0, (4- 1))]['c_1'])            
            
    def train_full(self, df_x2, df_y2, epoch, batch_sz):
        self.iter_ = int(df_x2.shape[0] / batch_sz)
        print('Iters: %s. Epoch:  %s' % (self.iter_, epoch))
        for e in range(epoch):
            e_loss = 0
            for i in range(self.iter_):
                iter_dict = {self.x_i: df_x2[(i * batch_sz):((i + 1) * batch_sz)],
                             self.y_i: df_y2[(i * batch_sz):((i + 1) * batch_sz)]}
                sess.run(self.optimiser, feed_dict=iter_dict)
                i_loss= sess.run(self.loss, feed_dict=iter_dict)
                e_loss+= i_loss
            print('Epoch %s, loss %s' % (e, e_loss))
        return e_loss
    
    def tr_te_fullALT(self, df_x2, df_y2, epoch, batch_sz, df_xtest, df_ytest):
        self.iter_ = int(df_x2.shape[0] / batch_sz)
        print('Iters: %s. Epoch:  %s' % (self.iter_, epoch))
        for e in range(epoch):
            e_loss = 0
            for i in range(self.iter_):
                self.rand_input(p= (i/ self.iter_))
                iter_dict = {self.x_i: df_x2[(i * batch_sz):((i + 1) * batch_sz)],
                             self.y_i: df_y2[(i * batch_sz):((i + 1) * batch_sz)]}
                sess.run(self.optimiser, feed_dict= iter_dict)
                i_loss= sess.run(self.loss, feed_dict= iter_dict)
                e_loss+= i_loss
                self.rand_reset()
                
                if i== round((self.iter_/ 2),0):
                    self.chang_reset()
                    trai_dict = {self.x_i: df_x2,
                         self.y_i: df_y2} 
                    tr_loss= sess.run(self.loss, feed_dict= trai_dict)
                    test_dict = {self.x_i: df_xtest,
                         self.y_i: df_ytest} 
                    te_loss= sess.run(self.loss, feed_dict= test_dict)
                    print('Epoch %s.5, train loss %s, test loss %s' % (self.n_epoch, tr_loss, te_loss))
            
            self.chang_reset()
            self.n_epoch+= 1
            trai_dict = {self.x_i: df_x2,
                         self.y_i: df_y2} 
            tr_loss= sess.run(self.loss, feed_dict= trai_dict)
            test_dict = {self.x_i: df_xtest,
                         self.y_i: df_ytest} 
            te_loss= sess.run(self.loss, feed_dict= test_dict)
            print('Epoch %s, train loss %s, test loss %s' % (self.n_epoch, tr_loss, te_loss))
        return e_loss

    def tr_te_full(self, df_x2, df_y2, epoch, batch_sz, df_xtest, df_ytest):
        self.iter_ = int(df_x2.shape[0] / batch_sz)
        print('Iters: %s. Epoch:  %s' % (self.iter_, epoch))
        for e in range(epoch):
            e_loss = 0
            for i in range(self.iter_):
                iter_dict = {self.x_i: df_x2[(i * batch_sz):((i + 1) * batch_sz)],
                             self.y_i: df_y2[(i * batch_sz):((i + 1) * batch_sz)]}
                sess.run(self.optimiser, feed_dict= iter_dict)
                i_loss= sess.run(self.loss, feed_dict= iter_dict)
                e_loss+= i_loss
                
                if i== round((self.iter_/ 2), 0):
                    trai_dict = {self.x_i: df_x2,
                         self.y_i: df_y2} 
                    tr_loss= sess.run(self.loss, feed_dict= trai_dict)
                    test_dict = {self.x_i: df_xtest,
                         self.y_i: df_ytest} 
                    te_loss= sess.run(self.loss, feed_dict= test_dict)
                    print('Epoch %s.5, train loss %s, test loss %s' % (self.n_epoch, tr_loss, te_loss))
            
            self.n_epoch+= 1
            trai_dict = {self.x_i: df_x2,
                         self.y_i: df_y2} 
            tr_loss= sess.run(self.loss, feed_dict= trai_dict)
            test_dict = {self.x_i: df_xtest,
                         self.y_i: df_ytest} 
            te_loss= sess.run(self.loss, feed_dict= test_dict)
            print('Epoch %s, train loss %s, test loss %s' % (self.n_epoch, tr_loss, te_loss))
        return e_loss
    
    def save_model(self, sess, NN_name= 'default'):
        file_path= './'+ NN_name + '/'
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        saver = tf.train.Saver()
        saver.save(sess, file_path+ 'model.checkpoint')
        print('Model saved')

    def load_model(self, sess, NN_name= 'default'):
        file_path = './' + NN_name + '/'
        saver = tf.train.Saver()
        saver.restore(sess, file_path+ 'model.checkpoint')
        print('Model loaded')

model4= Seq_alt2DeepSeq4_NN(t_in= 24, t_out= 12, d_in= 95, d_out= 1
                   , n_units_e= 20, n_stacks_e= 3, n_units_d= 20
                   , starter_learning_rate= 0.00004
                   , reg_val= 0.0001)

model= RRN_2_MLP_NN(t_in= 24, t_out= 12, d_in= 95, d_out= 1
                   , n_units= 20
                   , starter_learning_rate= 0.0004
                   , reg_val= 0.04
                   , alt= 'Y')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    model.tr_te_full(x_tr, y_tr, 5, 50, x_te, y_te)
    model.save_model(sess, model.__class__.__name__[:-2]+ model.m_name+ '_01') 
    print(model.__class__.__name__[:-2]+ model.m_name+ '_01')



model= Enc_2_DecALT(t_in= 24, t_out= 12, d_in= 95, d_out= 1
                   , n_units_e= 20, n_stacks_e= 3, n_units_d= 20
                   , starter_learning_rate= 0.00001
                   , reg_val= 0.0001)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.tr_te_fullALT(x_tr, y_tr, 1, 50, x_te, y_te)
    model.save_model(sess, model.__class__.__name__[: -2]+ model.m_name+ '_01')
