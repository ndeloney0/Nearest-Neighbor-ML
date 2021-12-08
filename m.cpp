#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;

class GenerateFeatures {
public:
    bool ReadFile(string fileName) {
        ifstream fin(fileName);
        if (!fin.is_open()) {
            cout << "Error reading from file!" << endl;
            return false;
        }
        
        double num;
        vector<double> temp;
        while (fin >> num) {
            temp.push_back(num);
            if (fin.peek() == '\n') {
                data.push_back(temp);
                temp.clear();
            }
        }

        fin.close();
        return true;
    }

    void NormalizeData() {
        for (int i = 1; i < data[0].size(); i++) {
            double mean = 0, dev = 0;
            for (int j = 0; j < data.size(); j++) {
                mean += data[j][i];
            }
            mean /= data.size();
            // cout << mean << endl;
            for (int j = 0; j < data.size(); j++) {
                dev += pow((data[j][i] - mean), 2);
            }
            // cout << dev << endl;
            dev = sqrt(dev/(data.size() - 1));
            // cout << dev << endl;
            for (int j = 0; j < data.size(); j++) {
                data[j][i] = (data[j][i] - mean)/dev;
            }
        }

        for (int i = 1; i < data[0].size(); i++) {
            for (int j = 0; j < data.size(); j++) {
                cout << data[j][i] << " ";
            }
            cout << endl;
        }
    }

private:
    vector<vector<double> > data;
};

int main() {
    GenerateFeatures *g = new GenerateFeatures;
    string fileName = "e.txt";
    // cout << "Enter name of file: ";
    // getline(cin, fileName);
    if (!g->ReadFile(fileName)) {
        cout << "ok";
    }
    g->NormalizeData();

    return 0;
}