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
    public partial class Form6product : Form
    {
        OleDbConnection con = new OleDbConnection(@"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\user\Desktop\project.accdb");

        //int larg = 0;
        public static int sump = 0;
        public float s = 0;
        public Form6product()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
                Form5animals f5 = new Form5animals();
                this.Hide();
                f5.ShowDialog();


            
        }
        

        private void Form6product_Load(object sender, EventArgs e)
        {
            this.Size = MaximumSize;
        }

        private void button5_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox1.Text), i = 2;
            t += 1;
            s += i;
            textBox1.Text = t.ToString();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox2.Text), i = 25;
            t += 1;
            s += i;
            textBox2.Text = t.ToString();
        }

        private void button7_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox3.Text), i = 4;
            t += 1;
            s += i;
            textBox3.Text = t.ToString();
        }

        private void button9_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox4.Text), i = 15;
            t += 1;
            s += i;
            textBox4.Text = t.ToString();
        }

        private void button11_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox5.Text), i = 4;
            t += 1;
            s += i;
            textBox5.Text = t.ToString();
        }

        private void button13_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox6.Text), i = 12;
            t += 1;
            s += i;
            textBox6.Text = t.ToString();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox2.Text), i = 25;
            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox2.Text = t.ToString();
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox1.Text), i = 2;
            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox1.Text = t.ToString();
            }
        }

        private void button6_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox3.Text), i = 4;
            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox3.Text = t.ToString();
            }
        }

        private void button8_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox4.Text), i = 15;
            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox4.Text = t.ToString();
            }
        }

        private void button10_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox5.Text), i = 4;
            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox5.Text = t.ToString();
            }
        }

        private void button12_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox6.Text), i = 12;
            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox6.Text = t.ToString();
            }
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
                    cmd.Parameters.AddWithValue("@a", "honye");
                    cmd.Parameters.AddWithValue("@b", 25);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox2.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox2.Text) * 25);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox2.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "yoghart");
                    cmd.Parameters.AddWithValue("@b", 5);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox3.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox3.Text) * 5);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox3.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "oil");
                    cmd.Parameters.AddWithValue("@b", 15);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox4.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox4.Text) * 15);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox4.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "zaatar");
                    cmd.Parameters.AddWithValue("@b", 4);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox5.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox5.Text) * 4);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox5.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "gazelles");
                    cmd.Parameters.AddWithValue("@b", 25);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox2.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox2.Text) * 25);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox6.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "milk");
                    cmd.Parameters.AddWithValue("@b", 2);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox6.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox6.Text) * 2);
                    cmd.ExecuteNonQuery();
                }

                con.Close();
            }
        }

        private void Form6product_Load_1(object sender, EventArgs e)
        {
            //this.Size = MaximumSize;
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            Form5animals f5 = new Form5animals();
            this.Hide();
            f5.ShowDialog();
        }

        private void button1_Click_2(object sender, EventArgs e)
        {
            Form5animals f5 = new Form5animals();
            this.Hide();
            f5.ShowDialog();
        }

        private void pictureBox7_MouseMove(object sender, MouseEventArgs e)
        {
            //MessageBox.Show("");
        }
    }
}
