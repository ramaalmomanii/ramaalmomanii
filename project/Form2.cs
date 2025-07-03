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
    public partial class Form2 : Form
        
    {
        
        private OleDbConnection con = new OleDbConnection(@"Provider=Microsoft.Jet.OLEDB.4.0");
        public Form2()
        {
            InitializeComponent();

        }

        private void button1_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void textBox3_TextChanged(object sender, EventArgs e)
        {

        }

        private void Form2_Load(object sender, EventArgs e)
        {
            //this.Size = MaximumSize;
            cmbcity.SelectedIndex = 0;
            
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label6_Click(object sender, EventArgs e)
        {

        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {

        }

        private void button2_Click(object sender, EventArgs e)
        {
            try
            {
                if (txtpassword.Text == txtcp.Text && txtusername.Text!= "" && textBox1.Text != "" && textBox2.Text !="" )
                {
                    /*يعمل ابديت ع الداتابيز*******************
                    con.Open();
                    OleDbCommand cmd = new OleDbCommand("insert into project values(@a,@b)", con);
                    cmd.Parameters.AddWithValue("@a", txtusername.Text);
                    cmd.Parameters.AddWithValue("@b", txtpassword.Text);
                    cmd.ExecuteNonQuery();
                    MessageBox.Show("This user is updated pleas loge in ");
                    con.Close();
                    
                    Form1 f1 = new Form1();
                    this.Hide();
                    f1.ShowDialog();*/
                    projectDataSetTableAdapters.projectTableAdapter T = new projectDataSetTableAdapters.projectTableAdapter();
                    projectDataSet.projectDataTable signin = T.thysignin(txtusername.Text);

                    if (signin.Rows.Count > 0)
                    {
                        MessageBox.Show("error");

                    }
                    else
                    {
                        T.InsertQuery(txtusername.Text, txtpassword.Text);
                        MessageBox.Show("Your account is created now back to login ");
                        Form1 f1 = new Form1();
                        this.Hide();
                        f1.ShowDialog();

                    }

                }
                else
                {
                    MessageBox.Show("incorect password");
                }
            }
            catch (Exception e1)
            {
                MessageBox.Show("invaled username " + e1);
            }
            
        }
            

        private void rdomale_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void button3_Click(object sender, EventArgs e)
        {
            Form1 f1 = new Form1();
            this.Hide();
            f1.ShowDialog();
        }
    }
}
