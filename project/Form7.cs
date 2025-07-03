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
    public partial class Form7 : Form
    {

       private OleDbConnection con = new OleDbConnection(@"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\Users\user\Desktop\project.accdb");
        public Form7()
        {
            
            InitializeComponent();
        }

        private void Form7_Load(object sender, EventArgs e)
        {
            //this.Size = MaximumSize;
            fillgrid();
        }
        private void fillgrid()
         {
            try
            {
                con.Open(); 
                OleDbDataAdapter da = new OleDbDataAdapter("select * from bill",con);
                DataTable dt = new DataTable();
                da.Fill(dt);
                dataGridView1.DataSource = dt;
                
                con.Close();
            }
            catch (Exception)
            {
                MessageBox.Show("Error in selecting rows drom table");
            }
       }

        private void button3_Click(object sender, EventArgs e)
        {
            Form6product f6 = new Form6product();
            this.Hide();
            f6.ShowDialog();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            //total();
            fillgrid();
        }
        public int s = 0;
        private void button8_Click(object sender, EventArgs e)
        {
           //int s = 0;
            s = Form4plants.sump + Form5animals.sump + Form6product.sump;
            MessageBox.Show(s.ToString());
            textBox1.Text = s.ToString();
            total();

        }
        private void total()
        {
            con.Open();
            OleDbCommand cmd = new OleDbCommand("select sum(total price) from bill", con);
            cmd.ExecuteNonQuery();
            textBox1.Text = cmd.ToString();
            con.Close();
        }

        private void button1_Click(object sender, EventArgs e)
        {

        }

        private void button4_Click(object sender, EventArgs e)
        {
            MessageBox.Show("Welcom to  eFAWATEERcom to pay total price = " + s.ToString());
        }

        private void button5_Click(object sender, EventArgs e)
        {
            MessageBox.Show("It's Not working now, try another way . BUT YOR TOTAL PRICE = " + s.ToString());
        }

        private void button7_Click(object sender, EventArgs e)
        {
            MessageBox.Show("Welcom to PayPal to pay total price = " + s.ToString());
        }

        private void button6_Click(object sender, EventArgs e)
        {
            MessageBox.Show("It's Not working now, try another way . BUT YOR TOTAL PRICE = " + s.ToString());
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            con.Open();
            OleDbCommand cmd = new OleDbCommand("delete * from bill", con);
            cmd.ExecuteNonQuery();
            con.Close();
            zForm4 f4 = new zForm4();
            this.Hide();
            f4.ShowDialog();

        }
    }

}
