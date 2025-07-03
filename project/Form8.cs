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
    public partial class Form8 : Form
    {
        private OleDbConnection con = new OleDbConnection(@"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\user\Desktop\project.accdb");
        public Form8()
        {
            InitializeComponent();
        }

        private void Form8_Load(object sender, EventArgs e)
        {
            //this.Size = MaximumSize;

        }

        private void button1_Click(object sender, EventArgs e)
        {
            Form7 f7 = new Form7();
            this.Hide();
            f7.ShowDialog();

        }

        private void txtUsername_TextChanged(object sender, EventArgs e)
        {
            
        }
        private void show()
        {
            try
            {
                con.Open();
                OleDbDataAdapter da = new OleDbDataAdapter("select * from project", con);
                DataTable dt = new DataTable();
                da.Fill(dt);
                MessageBox.Show("Yor password is updated");
                con.Close();
            }
            catch (Exception)
            {
                MessageBox.Show("Error in changing your passeword");
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (txtop.Text == Form1.pass && txtnp1.Text == txtnp2.Text && txtnp1.Text !=txtop.Text)
            {
                try
                {
                    con.Open();
                    OleDbCommand cmd = new OleDbCommand("update project set password=@a where UserName=@b", con);
                    cmd.Parameters.AddWithValue("@a", txtop.Text);
                   // cmd.Parameters.AddWithValue("@u", textBox1.Text);
                    cmd.Parameters.AddWithValue("@b", Form1.User);
                    cmd.ExecuteNonQuery();
                    con.Close();
                    //show();
                    MessageBox.Show("Your password is updated");
                }
                catch (Exception)
                {
                    MessageBox.Show("you can't updated password");
                }

            }
            
            
        }

        private void button3_Click(object sender, EventArgs e)
        {
            Form1 f1 = new Form1();
            this.Hide();
            f1.ShowDialog();
        }
    }
}
