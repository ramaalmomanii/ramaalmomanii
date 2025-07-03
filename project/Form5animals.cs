using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Data.OleDb;

namespace project
{
    public partial class Form5animals : Form
    {
        //int larg = 0;
        public static int sump = 0;
        OleDbConnection con = new OleDbConnection(@"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\user\Desktop\project.accdb");
        public float s = 0;
        public Form5animals()

        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Form3 f3 = new Form3();
            this.Hide();
            f3.ShowDialog();
        }

        private void textBox1_MouseMove(object sender, MouseEventArgs e)
        {
            //textBox1.Text = Text;
        }

        private void textBox4_TextChanged(object sender, EventArgs e)
        {

        }

        private void button5_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox1.Text), i = 2500;
            t += 1;
            s += i;
            textBox1.Text = t.ToString();
        }

        private void button4_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox1.Text), i = 2500;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox1.Text = t.ToString();
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox2.Text), i = 5;
            t += 1;
            s += i;
            textBox2.Text = t.ToString();
        }

        private void button13_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox6.Text), i = 20;
            t += 1;
            s += i;
            textBox6.Text = t.ToString();
        }

        private void button11_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox5.Text), i = 990;
            t += 1;
            s += i;
            textBox5.Text = t.ToString();
        }

        private void button9_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox4.Text), i = 1500;
            t += 1;
            s += i;
            textBox4.Text = t.ToString();
        }

        private void button7_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox3.Text), i = 350;
            t += 1;
            s += i;
            textBox3.Text = t.ToString();
        }

        private void button14_Click(object sender, EventArgs e)
        {
            sump = int.Parse(s.ToString());
            insertrow();
            MessageBox.Show("your Total price " + s.ToString());
            if (s > 0)
            {
                Form7 f7 = new Form7();
                this.Hide();
                f7.ShowDialog();
            }
        }

        private void button12_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox6.Text), i = 20;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox6.Text = t.ToString();
            }
        }

        private void button10_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox5.Text), i = 990;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox5.Text = t.ToString();
            }
        }

        private void button8_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox4.Text), i = 1500;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox4.Text = t.ToString();
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox2.Text), i = 15;
            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox2.Text = t.ToString();
            }
        }

        private void Form5animals_Load(object sender, EventArgs e)
        {
            //this.Size = MaximumSize;
        }

        private void button6_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox3.Text), i = 350;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox3.Text = t.ToString();
            }
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            insertrow();
            Form3 f4 = new Form3();
            this.Hide();
            f4.ShowDialog();
        }
        private void insertrow()
        {
            if (textBox1.Text == textBox2.Text && textBox2.Text == textBox3.Text && textBox3.Text == textBox4.Text && textBox5.Text == textBox4.Text && textBox6.Text == textBox5.Text && textBox6.Text == "0")
            {

            }
            else
            {

                con.Open();
                if (int.Parse(textBox1.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "horse");
                    cmd.Parameters.AddWithValue("@b", 2500);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox1.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox1.Text) * 2500);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox2.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "cheken");
                    cmd.Parameters.AddWithValue("@b", 5);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox2.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox2.Text) * 5);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox3.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "sheep");
                    cmd.Parameters.AddWithValue("@b", 350);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox3.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox3.Text) * 350);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox4.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "cow");
                    cmd.Parameters.AddWithValue("@b", 1500);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox4.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox4.Text) * 1500);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox5.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "gazelles");
                    cmd.Parameters.AddWithValue("@b", 990);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox5.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox5.Text) * 990);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox6.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "rabbit");
                    cmd.Parameters.AddWithValue("@b", 20);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox6.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox6.Text) * 20);
                    cmd.ExecuteNonQuery();
                }

                con.Close();
            }
        }

        private void button15_Click(object sender, EventArgs e)
        {
            insertrow();
            Form7 f4 = new Form7();
            this.Hide();
            f4.ShowDialog();
        }
    }

}
