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
    
    public partial class Form3 : Form
    {
        //string textFromFirstForm = ((Form1)Application.OpenForms["Form1"]).t
       // Form1 f1 = new Form1();
        //string username;
        private OleDbConnection con = new OleDbConnection(@"Provider=Microsoft.Jet.OLEDB.4.0");
        private OleDbConnection showpr = new OleDbConnection(@"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\user\Desktop\project.accdb");
        public Form3()
        {
            InitializeComponent();
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            if (chdanimals.Checked)
            {
                Form5animals f5 = new Form5animals();
                this.Hide();
                f5.ShowDialog();

            }
        }

        private void chdplants_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void treeView1_AfterSelect(object sender, TreeViewEventArgs e)
        {

        }

        private void Form3_Load(object sender, EventArgs e)
        {
            //this.Size = MaximumSize;
            /*TreeNode animal;
            animal = treeView1.Nodes.Add("hgh");
            animal = treeView1.Nodes.Add("98t");
            animal.Nodes[0].Nodes.Add("cow");
            animal.Nodes[0].Nodes.Add("rabit");
            animal.Nodes[0].Nodes.Add("cat");
            animal.Nodes[0].Nodes.Add("dog");
            animal.Nodes[0].Nodes.Add("غزال");
            animal.Nodes[1].Nodes.Add("عصفور كناري");*/ 

        }
        
        
        

        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            //string s = "jkgfakhashfghdfbsdfdsd" ;
            //lblplants.Text = s;
        }

        private void pictureBox1_MouseLeave(object sender, EventArgs e)
        {
            //lblplants.Text = "";
        }

        private void chdanimals_CheckedChanged(object sender, EventArgs e)
        {
            if ( chdanimals.Checked)
            {
                Form5animals f5 = new Form5animals();
                this.Hide();
                f5.ShowDialog();

            }
        }

        private void checkBox1_CheckedChanged_1(object sender, EventArgs e)
        {
            if (chdplant.Checked)
            {
                Form4plants f4 = new Form4plants();
                this.Hide();
                f4.ShowDialog();

            }
        }

        private void chdproducts_CheckedChanged(object sender, EventArgs e)
        {
            if (chdproducts.Checked)
            {
                Form6product f6 = new Form6product();
                this.Hide();
                f6.ShowDialog();

            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void deletMyAccountToolStripMenuItem_Click(object sender, EventArgs e)
        {
            //delet acount
            this.BackColor = Color.DarkGreen;
            


        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Form8 f = new Form8();
            this.Hide();
            f.ShowDialog();
        }

        private void changePasswordToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Form8 f = new Form8();
            this.Hide();
            f.ShowDialog();
            
        }

        private void myAccountToolStripMenuItem_Click(object sender, EventArgs e)
        {
           
        }

        private void dataGridView1_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {
            Form8 f8 = new Form8();
            this.Hide();
            f8.ShowDialog();

        }

        private void menuStrip1_ItemClicked(object sender, ToolStripItemClickedEventArgs e)
        {

        }
        private void show()
        {
            try
            {
                showpr.Open();
                OleDbDataAdapter sh = new OleDbDataAdapter("select * from proudect ", showpr);
                DataTable dtsh = new DataTable();
                sh.Fill(dtsh);
                dataGridView2.DataSource = dtsh;
                //dtsh.ex
                showpr.Close();

            }
            catch (Exception)
            {
                MessageBox.Show("Error ");
            }
        }
        private void button2_Click(object sender, EventArgs e)
        {
            dataGridView2.Visible = true;

            show(); 
        }

        private void exitToolStripMenuItem2_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void logoutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Form1 f1 = new Form1();
            this.Hide();
            f1.ShowDialog();
        }
    }
}
