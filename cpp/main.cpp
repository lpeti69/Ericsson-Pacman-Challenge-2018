#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <map>

using namespace std;

//////////////////////////////////////////////////////////////////////////// MAP
////////////////////////////////////////////////////////////////////////////////
class Map {
public:
    const size_t x;			// map width
    const size_t y;			// map height
	//
    Map(const size_t x, const size_t y)
	: x(x), y(y), M(y) {  }
	void read(istream& in) {
		for (size_t i = 0; i < y; ++i)
			getline(in, M[i]);
	}
    string operator[] (const size_t i) const {
		return M[i];
    }
private:
    vector<string> M;
};
istream& operator>>(istream& in, Map& M) {
	M.read(in);
	return in;
}

///////////////////////////////////////////////////////////////////////// PACKAM
////////////////////////////////////////////////////////////////////////////////
class Pacman {
public:
	size_t id;				// pacman ID (index)
	string team;			// team name
	size_t x;				// x position on the map
	size_t y;				// y position on the map
	size_t boost_remain;	// remained boost ticks
	size_t points;			// earned points
	string plus_points;		// earned pluspoints
	string prev_cmd;		// previous commands
	//
	Pacman() {  }
	void read(istream& in) {
		string line;
		getline(in, line);
		stringstream ss(line);
		ss >> id >> team >> y >> x
		   >> boost_remain >> points >> plus_points;
		if (getline(ss, line)) prev_cmd = line.substr(1);
		else prev_cmd = "";	
	}
private:
	//
};
istream& operator>>(istream& in, Pacman& P) {
	P.read(in);
	return in;
}

////////////////////////////////////////////////////////////////////////// GHOST
////////////////////////////////////////////////////////////////////////////////
class Ghost {
public:
	char id;				// ghost ID (type)
	size_t x;				// x position on the map
	size_t y;				// y position on the map
	size_t eatable;			// remained eatable ticks
	size_t frozen;			// remained frozen ticks
	//
	Ghost() {  }
	void read(istream& in) {
		in >> id >> y >> x >> eatable >> frozen;
	}
private:
	//
};
istream& operator>>(istream& in, Ghost& H) {
	H.read(in);
	return in;
}

/////////////////////////////////////////////////////////////////////////// GAME
////////////////////////////////////////////////////////////////////////////////
class Game {
public:
	static const size_t max_ticks = 480;
	//
	const int id;			// game ID
	int pid;				// pacman ID
	int tick;				// current tick
	vector<string> msgs;	// all messages from line 2
	//
	Map M;					// map
	vector<Pacman> P;		// packmans
	vector<Ghost> H;		// ghosts
	//
	Game(const int g, const int t, const int p, const size_t x, const size_t y)
	: id(g), tick(t), pid(p), M(x, y) {  }
private:
	//
};




////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
map<int,Game> GAMES;
Game& read() {
	string line, endline;
	
	// first line
	int g, t, p;
	cin >> g >> t >> p;
	// second line
	size_t x, y, Np, Ng;
	string msg("");
	getline(cin, endline);
	getline(cin, line);
	stringstream ss(line);
	ss >> y >> x >> Np >> Ng;
	if (getline(ss, msg)) {
		msg = msg.substr(1);
		cerr << msg << endl;
	}
	
	// game object
	if (!GAMES.count(g)) {
		GAMES.emplace(
			piecewise_construct,
			forward_as_tuple(g),
			forward_as_tuple(g, t, p, x, y));
	}
	Game& G = GAMES.find(g)->second;
	
	// meta
	G.pid = p;
	G.tick = t;
	G.msgs.emplace_back(msg);
	
	// map
	cin >> G.M;
	
	// packmans
	G.P.resize(Np);
	for (int i = 0; i < Np; ++i) {
		cin >> G.P[i];
	}
	
	//gosts
	G.H.resize(Ng);
	for (int i = 0; i < Ng; ++i) {
		cin >> G.H[i];
	}
	
	return G;
}




////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main() {
	ios_base::sync_with_stdio(false);
	
	while (true) {
		Game& G = read();
		if (G.pid == -1) {
			GAMES.erase(G.id);
			if (GAMES.size() == 0) 
				break;
		}
		//
		cout << G.id << ' ' << G.tick << ' ' << ">" << endl;
	}
	
	
}