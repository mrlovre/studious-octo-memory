import tensorflow as tf
import numpy as np


def main():
    x = tf.placeholder(tf.float32, [None])
    y_ = tf.placeholder(tf.float32, [None])
    a = tf.Variable(0.)
    b = tf.Variable(0.)

    x_values = np.arange(5)
    y__values = 2 * x_values - 1 + np.random.randn(5)

    y = a * x + b

    analytic_grad_a = 2 * tf.reduce_sum((y - y_) * x)
    analytic_grad_b = 2 * tf.reduce_sum(y - y_)

    loss = tf.reduce_sum((y - y_) ** 2)

    trainer = tf.train.GradientDescentOptimizer(1e-3)
    train_op = trainer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for i in range(100):
        a_grad, b_grad = gradients = trainer.compute_gradients(loss, [a, b])
        val_loss, _, val_a, val_b, val_a_grad, val_b_grad, val_analytic_a_grad, val_analytic_b_grad = \
            sess.run([loss, train_op, a, b, a_grad[0], b_grad[0], analytic_grad_a, analytic_grad_b],
                     feed_dict={x: x_values, y_: y__values})
        # print(a_grad[0], b_grad[0])
        # trainer.apply_gradients(gradients)
        print(i, val_loss, val_a, val_b)
        print('  tf gradients:', val_a_grad, val_b_grad)
        print('  analytic gradients:', val_analytic_a_grad, val_analytic_b_grad)


if __name__ == '__main__':
    main()
