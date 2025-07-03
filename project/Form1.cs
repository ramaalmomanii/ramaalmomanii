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
    public partial class Form1 : Form
    {
        public static string pass = "";
        public static string User ="";
        private OleDbConnection con = new OleDbConnection(@"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\user\Desktop\AL-MOMANI2010\project\project2.accdb");
        public Form1()
        {
InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {

        }            

        private void Form1_Load(object sender, EventArgs e)
        {
            

            
            //this.Size = MaximumSize;
            //Label.parent = PictureBox;
            //this.BackgroundImage = Image.FromFile(@"");
            //pictureBox1.Image = Image.FromFile(@"C:\Users\user\Desktop\photo_2023-05-15_18-32");
            //pictureBox1.Size = this.MaximumSize;
            
        }

        

        private void pictureBox1_Click(object sender, EventArgs e)
        {
            
        }

        private void button1_Click_1(object sender, EventArgs e)
        {

        }

        private void label3_Click(object sender, EventArgs e)
        {
            Form2 f2 = new Form2();
            this.Hide();
            f2.ShowDialog();
        }

        private void button2_Click(object sender, EventArgs e)
        {

        }

        private void button2_Click_1(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void button1_Click_2(object sender, EventArgs e)
        {

            try
            {
                
                projectDataSetTableAdapters.projectTableAdapter user = new projectDataSetTableAdapters.projectTableAdapter();
                //projectDataSetTableAdapters.projectTableAdapter u = new projectDataSetTableAdapters.projectTableAdapter();
                //projectDataSet.projectDataTable pt = u.GetDataBy1(txtUsername.Text, txtpassword.Text);
                projectDataSet.projectDataTable pt = user.GetDataBy(txtUsername.Text, txtpassword.Text);
               
                if (pt.Rows.Count == 1)
                {
                    pass = txtpassword.Text;
                    User = txtUsername.Text;
                    Form3 f3 = new Form3();
                    this.Hide();
                    MessageBox.Show("Welcame " + txtUsername.Text + " to our farm ");
                    f3.ShowDialog();
                   
                }
                else
                {
                    txtpassword.Text = "";
                    MessageBox.Show("username or password incorrect");
                    
                }
            }
            catch(Exception e2)

            {
                MessageBox.Show("Error " + e2);
            }
            
        }

        private void button3_Click(object sender, EventArgs e)
        {
            
        }

        private void button3_Click_1(object sender, EventArgs e)
        {/*
            Form3 f3 = new Form3();
            this.Hide();
            f3.ShowDialog();*/
        }

        private void label2_Click(object sender, EventArgs e)
        {

        }
    }
}
