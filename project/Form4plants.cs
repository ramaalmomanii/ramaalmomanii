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
    
    public partial class Form4plants : Form
    {
        //int larg = 0;
        public static int sump = 0, a=1;

        private OleDbConnection con = new OleDbConnection(@"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\user\Desktop\project.accdb");
        public float s = 0;

        //c1=15 , c2=3 , c3=5 , c4=2 , c5=2 , c6=6; 
        public Form4plants()
        {
            InitializeComponent();

        }

        private void Form4_Load(object sender, EventArgs e)
        {
            //this.Size = MaximumSize;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            insertrow();
            Form3 f3 = new Form3();
            this.Hide();
            f3.ShowDialog();
        }

        private void checkBox3_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void button6_Click(object sender, EventArgs e)
        {
        }


        private void button13_Click(object sender, EventArgs e)
        {
            
            

            float t = float.Parse(textBox1.Text), i = 15;
            t += 1;
            s += i;
            sump += int.Parse(s.ToString());
            textBox1.Text = t.ToString();
            /*val += 15;
            textBox1.Text = val.ToString();
            s += val;
            MessageBox.Show("FINAL PRISE" + s.ToString());*/


        }

        private void button12_Click(object sender, EventArgs e)
        {

            float t = float.Parse(textBox1.Text), i = 15;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                sump -= int.Parse(s.ToString());
                textBox1.Text = t.ToString();

            }
        }

        private void button11_Click(object sender, EventArgs e)
        {

        }

        private void button10_Click(object sender, EventArgs e)
        {


        }

        private void button8_Click(object sender, EventArgs e)
        {

        }

        private void button6_Click_1(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void c1_CheckedChanged(object sender, EventArgs e)
        {
            button12.Enabled = button13.Enabled = true;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox2.Text), i = 3;
            t += 1;
            s += i;
            sump += int.Parse(s.ToString());
            textBox2.Text = t.ToString();
        }

        private void button2_Click(object sender, EventArgs e)
        {

            float t = float.Parse(textBox2.Text), i = 3;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox2.Text = t.ToString();
                sump -= int.Parse(s.ToString());
            }
        }

        private void button7_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox4.Text), i = 10;
            t += 1;
            s += i;
            sump += int.Parse(s.ToString());
            textBox4.Text = t.ToString();
        }

        private void button9_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox5.Text), i = 10;
            t += 1;
            s += i;
            textBox5.Text = t.ToString();
            sump += int.Parse(s.ToString());
        }

        private void button11_Click_1(object sender, EventArgs e)
        {
            float t = float.Parse(textBox6.Text), i = 7;
            t += 1;
            s += i;
            textBox6.Text = t.ToString();
            sump += int.Parse(s.ToString());
        }

        private void button6_Click_2(object sender, EventArgs e)
        {

            float t = float.Parse(textBox4.Text), i = 10;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox4.Text = t.ToString();
                sump -= int.Parse(s.ToString());
            }
        }

        private void button8_Click_1(object sender, EventArgs e)
        {

            float t = float.Parse(textBox5.Text), i = 10;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox5.Text = t.ToString();
            }
        }

        private void button10_Click_1(object sender, EventArgs e)
        {

            float t = float.Parse(textBox6.Text), i = 7;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox6.Text = t.ToString();
                sump -= int.Parse(s.ToString());
            }
        }


        private void button4_Click(object sender, EventArgs e)
        {

            float t = float.Parse(textBox3.Text), i = 6;

            if (t > 0)
            {
                t -= 1;
                s -= i;
                textBox3.Text = t.ToString();
                sump -= int.Parse(s.ToString());
            }
        }

        private void c3_CheckedChanged(object sender, EventArgs e)
        {
            button5.Enabled = button4.Enabled = true;
        }

        private void c2_CheckedChanged(object sender, EventArgs e)
        {
            button2.Enabled = button3.Enabled = true;
        }

        private void c6_CheckedChanged(object sender, EventArgs e)
        {
            button11.Enabled = button10.Enabled = true;
        }

        private void c5_CheckedChanged(object sender, EventArgs e)
        {
            button8.Enabled = button9.Enabled = true;
        }

        private void c4_CheckedChanged(object sender, EventArgs e)
        {
            button6.Enabled = button7.Enabled = true;
        }

        private void button5_Click(object sender, EventArgs e)
        {
            float t = float.Parse(textBox3.Text), i = 6;
            t += 1;
            s += i;
            sump = int.Parse(s.ToString());
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
                   
                    cmd.Parameters.AddWithValue("@a", "large tree");
                    cmd.Parameters.AddWithValue("@b", 15);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox1.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox1.Text)*15);
                    cmd.ExecuteNonQuery();

                }
                if (int.Parse(textBox2.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    cmd.Parameters.AddWithValue("@a", "small cactus");
                    cmd.Parameters.AddWithValue("@b", 3);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox2.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox1.Text) * 3);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox3.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                   
                    cmd.Parameters.AddWithValue("@a", "small ornamental plant");
                    cmd.Parameters.AddWithValue("@b", 6);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox3.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox1.Text) * 6);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox4.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    
                    cmd.Parameters.AddWithValue("@a", "red flower");
                    cmd.Parameters.AddWithValue("@b", 10);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox4.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox1.Text) * 10);
                    cmd.ExecuteNonQuery();
                }
                if (int.Parse(textBox5.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    
                    cmd.Parameters.AddWithValue("@a", "pink flower");
                    cmd.Parameters.AddWithValue("@b", 10);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox5.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox1.Text) * 10);
                    cmd.ExecuteNonQuery();
                   
                }
                if (int.Parse(textBox6.Text) > 0)
                {
                    OleDbCommand cmd = new OleDbCommand("insert into bill values(@a,@b,@c,@d)", con);
                    
                    cmd.Parameters.AddWithValue("@a", "ornamental plant");
                    cmd.Parameters.AddWithValue("@b", 7);
                    cmd.Parameters.AddWithValue("@c", int.Parse(textBox6.Text));
                    cmd.Parameters.AddWithValue("@d", int.Parse(textBox1.Text) * 7);
                    cmd.ExecuteNonQuery();
                }

                con.Close();
            }
        }
    }
}