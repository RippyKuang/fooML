#include <autograd.h>
namespace autograd
{

    int Variable::_name = 0;
    std::vector<TapeEntry> Variable::gradient_tape;

    Variable Variable::operator+(const Variable &rhs)
    {
        return Variable(var + rhs.var);
    }

    Variable Variable::operator-(const Variable &rhs)
    {
        return Variable(var - rhs.var);
    }

    Variable Variable::operator*(const Variable &rhs)
    {
        Variable r(var * rhs.var);
        std::cout << r.name << "=" << name << "*" << rhs.name << std::endl;
        std::vector<std::string> inputs{name, rhs.name};
        std::vector<std::string> outputs{r.name};
        auto f = std::bind(mul_propagate, std::ref(*this), std::ref(rhs), std::placeholders::_1);
        gradient_tape.push_back(TapeEntry{inputs, outputs, f});
        return r;
    }

    std::vector<ATen::Matrix<float>> mul_propagate(const Variable &self, const Variable &rhs, std::vector<ATen::Matrix<float>> &dL_doutputs)
    {
        ATen::Matrix<float> dL_dr = dL_doutputs[0];
        ATen::Matrix<float> dr_dself = rhs.get_var();
        ATen::Matrix<float> dr_drhs = self.get_var();

        std::vector<ATen::Matrix<float>> dL_dinputs{dL_dr * dr_dself, dL_dr * dr_drhs};
        return dL_dinputs;
    }

    Variable Variable::operator/(const Variable &rhs)
    {
        Variable r(var / rhs.var);
        std::cout << r.name << " " << name << "/" << rhs.name << std::endl;
        std::vector<std::string> inputs{name, rhs.name};
        std::vector<std::string> outputs{r.name};
        auto f = std::bind(div_propagate, std::ref(*this), std::ref(rhs), std::placeholders::_1);
        gradient_tape.push_back(TapeEntry{inputs, outputs, f});
        return r;
    }

    std::vector<ATen::Matrix<float>> div_propagate(const Variable &self, const Variable &rhs, std::vector<ATen::Matrix<float>> &dL_doutputs)
    {
        ATen::Matrix<float> dL_dr = dL_doutputs[0];
        ATen::Matrix<float> dr_dself = 1.0f / rhs.get_var();
        ATen::Matrix<float> dr_drhs = (self.get_var() / (rhs.get_var() * rhs.get_var())) * (-1);

        std::vector<ATen::Matrix<float>> dL_dinputs{dL_dr * dr_dself, dL_dr * dr_drhs};
        return dL_dinputs;
    }

    void grad(Variable L, std::vector<Variable> desired_results)
    {
        int r,c;
        std::map<std::string, ATen::Matrix<float>> dL_d;
        std::tie(r, c) = L.var.getShape();
        dL_d[L.name] = ATen::Matrix<float>(r, c, 1);
        for (auto it = Variable::gradient_tape.rbegin(); it != Variable::gradient_tape.rend(); ++it)
        {
            auto entry = *it;
          
            std::vector<ATen::Matrix<float>> dL_doutputs = gather_grad(dL_d, entry.outputs);

            if (dL_doutputs.size() == 0)
                continue;
            std::vector<ATen::Matrix<float>> dL_dinputs = entry.propagate(dL_doutputs);

            for (int i = 0; i < dL_dinputs.size(); i++)
            {
                if (dL_d.count(entry.inputs[i]) != 0)
                    dL_d[entry.inputs[i]] = dL_d[entry.inputs[i]] + dL_dinputs[i];
                else
                    dL_d[entry.inputs[i]] = dL_dinputs[i];
            }
        }
        for (auto [key, val] : dL_d)
        {
            std::cout << "d" << L.name << "_d" << key << "=\n" << val << "\n"
                      << std::endl;
        }
    }

    std::vector<ATen::Matrix<float>> gather_grad(std::map<std::string, ATen::Matrix<float>> &dL_d, std::vector<std::string> &entries)
    {
        std::vector<ATen::Matrix<float>> dL_doutputs;
        for (auto key : entries)
            if (dL_d.count(key) != 0)
                dL_doutputs.push_back(dL_d[key]);   
        return dL_doutputs;
    }
}