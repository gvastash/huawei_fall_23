#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <stdint.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include <vector>
#include <list>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <deque>
#include <string>

#include <algorithm>
#include <functional>
#include <bitset>
#include <functional>
#include <chrono>
#include <random>
#include <memory>
#include <array>
#include <numeric>
#include <thread>

typedef int ll;

typedef unsigned long long int ull;

typedef float ld;

using namespace std;
using namespace std::chrono;

const ll mod1 = 1'000'000'000ll + 7;
const ll mod2 = 998'244'353ll;

//const ll inf = 1'000'000'000ll + 7;
//const ll inf = 1'100'000'000'000'000'000ll + 7;

const ld eps = 0.000001;

#pragma warning(disable: 4101)

auto StartTm = std::chrono::steady_clock::now();

ll CheckTL() {
    auto now = std::chrono::steady_clock::now();
    auto duration = duration_cast<milliseconds>(now - StartTm).count();

    return duration > 1'500;
}

struct TRandom {
    using Type = ll;
    mt19937 Rng;
    uniform_int_distribution<Type> Dist;

    TRandom() : Rng((int)(chrono::steady_clock::now().time_since_epoch().count())), Dist(0ll, std::numeric_limits<Type>::max()) {
//    TRandom() : Rng(1), Dist(0ll, std::numeric_limits<Type>::max()) {
    }

    Type GetRandomNumber() {
        return Dist(Rng);
    }
};

TRandom Rng;

void AllocateMemory(ld x) {
    ll size = x * 1024 * 1024;
    char* mem = new char[size];

//    for (ll i = 0; i < size; i += 8) {
//        *(ll*)(mem + i) = Rng.GetRandomNumber();
//    }
}


ll N, K, T, R;
vector<vector<vector<vector<ld>>>> S;
vector<vector<vector<vector<ld>>>> D;
ll J;

vector<tuple<ll, ll, ll, ll, ll>> F;
vector<vector<ll>> Frames;

vector<vector<vector<vector<ld>>>> P;
vector<ld> Completeness;
vector<vector<map<ll, set<ll>>>> RadioUsage;

vector<vector<vector<vector<ld>>>> Nominator;
vector<vector<vector<vector<ll>>>> Validity;

ll Tm = 1;

vector<vector<vector<vector<ld>>>> BestP;
vector<ld> BestCompleteness;
vector<vector<map<ll, set<ll>>>> BestRadioUsage;
vector<ll> BestRes;
ll Res;

ld AvgDuration;
ld AvgD;

ld InspectDuration();
ld InspectD();

ld Eps = 0.00001;

void Reset() {
    P = vector<vector<vector<vector<ld>>>>(T, vector<vector<vector<ld>>>(K, vector<vector<ld>>(R, vector<ld>(N))));
    Completeness = vector<ld>(J);
    RadioUsage = vector<vector<map<ll, set<ll>>>>(T, vector<map<ll, set<ll>>>(R));

    Nominator = vector<vector<vector<vector<ld>>>>(T, vector<vector<vector<ld>>>(K, vector<vector<ld>>(R, vector<ld>(N))));
    Validity = vector<vector<vector<vector<ll>>>>(T, vector<vector<vector<ll>>>(K, vector<vector<ll>>(R, vector<ll>(N))));
}

void Init() {
    // r + k * R + t * K * R
    Reset();

    Frames = vector<vector<ll>>(T);
    for (ll j = 0; j < J; j++) {
        auto [fid, size, n, start, duration] = F[j];
        for (ll t = start; t < start + duration; t++) {
            Frames[t].push_back(j);
        }
    }

    AvgDuration = InspectDuration();
    AvgD = -InspectD();

    BestP = P;
    BestCompleteness = Completeness;
    BestRadioUsage = RadioUsage;
    BestRes = vector<ll>(T);
}

void ReadInput(istream& in) {
    in >> N >> K >> T >> R;

    // r + k * R + t * K * R
    S = vector<vector<vector<vector<ld>>>>(T, vector<vector<vector<ld>>>(K, vector<vector<ld>>(R, vector<ld>(N))));
    for (ll t = 0; t < T; t++) {
        for (ll k = 0; k < K; k++) {
            for (ll r = 0; r < R; r++) {
                for (ll n = 0; n < N; n++) {
                    in >> S[t][k][r][n];
                }
            }
        }
    }

    // m + r * N + k * R * N
    D = vector<vector<vector<vector<ld>>>>(K, vector<vector<vector<ld>>>(R, vector<vector<ld>>(N, vector<ld>(N))));
    for (ll k = 0; k < K; k++) {
        for (ll r = 0; r < R; r++) {
            for (ll m = 0; m < N; m++) {
                for (ll n = 0; n < N; n++) {
                    in >> D[k][r][m][n];
                }
            }
        }
    }

    in >> J;
    F = vector<tuple<ll, ll, ll, ll, ll>>(J);
    for (ll j = 0; j < J; j++) {
        ll fid, size, uid, start, duration;
        in >> fid >> size >> uid >> start >> duration;
        F[j] = {fid, size, uid, start, duration};
    }

    Init();
}

void WriteOutput() {
    stringstream ss;
    ss.precision(6);
    ss.setf(ios::fixed);

    for (ll t = 0; t < T; t++) {
        for (ll k = 0; k < K; k++) {
            for (ll r = 0; r < R; r++) {
                for (ll n = 0; n < N; n++) {
                    if (P[t][k][r][n] == 0.0) {
                        ss << "0";
                    } else {
                        ss << P[t][k][r][n];
                    }
                    if (n == N - 1) {
                        ss << endl;
                    } else {
                        ss << " ";
                    }
                }
            }
        }
    }

    cout << ss.str();
}

void GenerateInput() {
    N = 1 + Rng.GetRandomNumber() % 100;
    K = 1 + Rng.GetRandomNumber() % 10;
    T = 1 + Rng.GetRandomNumber() % 1000;
    R = 1 + Rng.GetRandomNumber() % 10;

    // r + k * R + t * K * R
    S = vector<vector<vector<vector<ld>>>>(T, vector<vector<vector<ld>>>(K, vector<vector<ld>>(R, vector<ld>(N))));
    for (ll t = 0; t < T; t++) {
        for (ll k = 0; k < K; k++) {
            for (ll r = 0; r < R; r++) {
                for (ll n = 0; n < N; n++) {
                    S[t][k][r][n] = (ld)(Rng.GetRandomNumber() % 1'000'000) / 100;
                }
            }
        }
    }

    // m + r * N + k * R * N
    D = vector<vector<vector<vector<ld>>>>(K, vector<vector<vector<ld>>>(R, vector<vector<ld>>(N, vector<ld>(N))));
    for (ll k = 0; k < K; k++) {
        for (ll r = 0; r < R; r++) {
            for (ll m = 0; m < N; m++) {
                for (ll n = m + 1; n < N; n++) {
                    D[k][r][n][m] = D[k][r][m][n] = -(ld)(Rng.GetRandomNumber() % 1'000'000) / 500'000;
                }
            }
        }
    }

    J = 1 + Rng.GetRandomNumber() % 5'000;
    J = (J / N + 1) * N;

//    F = vector<tuple<ll, ll, ll, ll, ll>>(J);
//    for (ll j = 0; j < J; j++) {
//        ll fid = j;
//        ll size = Rng.GetRandomNumber() % 100'001;
//        ll uid = Rng.GetRandomNumber() % N;
//        ll start = Rng.GetRandomNumber() % T;
//        ll duration = 1 + Rng.GetRandomNumber() % (T - start);
//        F[j] = {fid, size, uid, start, duration};
//    }

    for (ll n = 0; n < N; n++) {
        set<ll> points;
        for (ll i = 0; points.size() < 2 * J / N && i < N * J; i++) {
            points.insert(Rng.GetRandomNumber() % T);
        }

        ll l = -1;
        for (auto x : points) {
            if (l == -1) {
                l = x;
                continue;
            }

            ll size = Rng.GetRandomNumber() % 100'001;
            ll uid = n;
            ll start = l;
            ll duration = x - l;

            F.push_back({0ll, size, uid, start, duration});
            l = -1;
        }
    }
    shuffle(F.begin(), F.end(), Rng.Rng);

    for (ll j = 0; j < J; j++) {
        get<0>(F[j]) = j;
    }

    Init();
}

void GenerateInputLowDuration(ll t, ll k, ll r, ll n, ll j) {
    T = t;
    K = k;
    R = r;
    N = n;
    J = j;

    // r + k * R + t * K * R
    S = vector<vector<vector<vector<ld>>>>(T, vector<vector<vector<ld>>>(K, vector<vector<ld>>(R, vector<ld>(N))));
    for (ll t = 0; t < T; t++) {
        for (ll k = 0; k < K; k++) {
            for (ll r = 0; r < R; r++) {
                for (ll n = 0; n < N; n++) {
                    S[t][k][r][n] = (ld)(Rng.GetRandomNumber() % 1'000'000) / 100;
                }
            }
        }
    }

    // m + r * N + k * R * N
    D = vector<vector<vector<vector<ld>>>>(K, vector<vector<vector<ld>>>(R, vector<vector<ld>>(N, vector<ld>(N))));
    for (ll k = 0; k < K; k++) {
        for (ll r = 0; r < R; r++) {
            for (ll m = 0; m < N; m++) {
                for (ll n = m + 1; n < N; n++) {
                    D[k][r][n][m] = D[k][r][m][n] = -(ld)(Rng.GetRandomNumber() % 1'000'000) / 500'000;
                    D[k][r][n][m] *= 0.1;
                    D[k][r][m][n] *= 0.1;
                }
            }
        }
    }

    vector<vector<ll>> frames(T);

    for (ll j = 0; j < J; j++) {
        ll t = Rng.GetRandomNumber() % T;
//        ll size = Rng.GetRandomNumber() % 100'001;
        ll size = Rng.GetRandomNumber() % 100;
        ll n = frames[t].size();
        N = max(n, N);
        frames[t].push_back(j);
        F.push_back({j, size, n, t, 1});
    }

    shuffle(F.begin(), F.end(), Rng.Rng);

    for (ll j = 0; j < J; j++) {
        get<0>(F[j]) = j;
    }

    Init();
}

ld RadioPowerLimit = 1.0;

ld CalcRadioPowerLeft(ll t, ll k, ll r) {
    ld res = 0;
    for (ll n = 0; n < N; n++) {
        res += P[t][k][r][n];
    }

//    return 4.0 - res;
    return RadioPowerLimit - Eps - res;
}

ld CalcCellPowerLeft(ll t, ll k) {
    ld res = 0;
    for (ll r = 0; r < R; r++) {
        for (ll n = 0; n < N; n++) {
            res += P[t][k][r][n];
        }
    }

    return (ld)R - Eps - res;
}

void SetPower(ll t, ll k, ll r, ll n, ll j, ld v) {
    Tm++;

    P[t][k][r][n] = v;
    if (v > 0.0) {
        RadioUsage[t][r][k].insert(j);
    } else {
        RadioUsage[t][r][k].erase(j);
        if (RadioUsage[t][r][k].empty()) {
            RadioUsage[t][r].erase(k);
        }
    }
}

ld CalcSignal(ll t, ll k, ll r, ll n) {
    ld nominator = log(S[t][k][r][n]) + log(P[t][k][r][n]);

    if (Validity[t][k][r][n] != Tm) {
        Nominator[t][k][r][n] = 0;
        for (auto j : Frames[t]) {
            auto [fid, size, m, start, duration] = F[j];

            if (m == n) {
                continue;
            }

            if (P[t][k][r][m] <= 0.0) {
                continue;
            }

            Nominator[t][k][r][n] += D[k][r][m][n];
            Validity[t][k][r][n] = Tm;
        }
    }

    nominator += Nominator[t][k][r][n];

    ld denominator = 1.0;

//    for (auto [dk, js] : RadioUsage[t][r]) {
//        if (dk == k) {
//            continue;
//        }
//
//        for (auto j : js) {
//            auto [fid, size, dn, start, duration] = F[j];
//
//            if (dn == n) {
//                continue;
//            }
//
//            ld logCur = log(S[t][dk][r][n]) + log(P[t][dk][r][dn]) - D[dk][r][n][dn];
//            denominator += exp(logCur);
//        }
//    }

    return nominator - log(denominator);
}


ld CalcSignal(ll t, ll k, ll n) {
    ld logRes = 0.0;

    ld cnt = 0;
    for (ll r = 0; r < R; r++) {
        if (P[t][k][r][n] <= 0.0) {
            continue;
        }

        logRes += CalcSignal(t, k, r, n);
        cnt += 1;
    }

    return logRes / cnt;
}

ld CalcSignal(ll t, ll n) {
    const ld W = 192;

    ld sum = 0;
    for (ll k = 0; k < K; k++) {
        ll cntUsesPower = 0;

        for (ll r = 0; r < R; r++) {
            cntUsesPower += (P[t][k][r][n] > 0.0 ? 1 : 0);
        }

        if (!cntUsesPower) {
            continue;
        }

        ld signal = CalcSignal(t, k, n);
        sum += cntUsesPower * log2(1 + exp(signal));
    }

    return W * sum;
}

void CheckPowerInvariant() {
    for (ll t = 0; t < T; t++) {
        for (ll k = 0; k < K; k++) {
            ld cellPower = 0.0;
            for (ll r = 0; r < R; r++) {
                ld cellRadioPower = 0.0;
                for (ll n = 0; n < N; n++) {
                    cellRadioPower += P[t][k][r][n];
                }
                if (cellRadioPower > 4.0) {
                    vector<tuple<ld, ll>> q;
                    for (ll n = 0; n < N; n++) {
                        q.push_back({P[t][k][r][n], n});
                    }
                    sort(q.begin(), q.end(), greater<>());
                    for (auto [v, n] : q) {
                        if (cellRadioPower <= 4.0) {
                            break;
                        }
                        ld nv = min((ld)v, (ld)(cellRadioPower - 4.0));
                        cellRadioPower -= nv;
                        P[t][k][r][n] -= nv;
                    }

                }
                cellPower += cellRadioPower;
            }
            if (cellPower > R) {
                vector<tuple<ld, ll, ll>> q;

                for (ll r = 0; r < R; r++) {
                    for (ll n = 0; n < N; n++) {
                        q.push_back({P[t][k][r][n], r, n});
                    }
                }

                sort(q.begin(), q.end(), greater<>());
                for (auto [v, r, n] : q) {
                    if (cellPower <= 4.0) {
                        break;
                    }
                    ld nv = min((ld)v, (ld)(cellPower - 4.0));
                    cellPower -= nv;
                    P[t][k][r][n] -= nv;
                }

            }
//            cerr << cellPower << endl;
        }
    }
}

bool IsCellAlreadyBooked(ll t, ll k, ll r) {
    for (ll n = 0; n < N; n++) {
        if (P[t][k][r][n] > 0.0) {
            return true;
        }
    }

    return false;
}

bool IsRadioAlreadyBooked(ll t, ll r, ll m) {
    for (ll k = 0; k < K; k++) {
        for (ll n = 0; n < N; n++) {
            if (n == m) {
                continue;
            }
            if (P[t][k][r][n] > 0.0) {
                return true;
            }
        }
    }
    return false;
}

bool IsTimeframeAlreadyUsed(ll t, ll n) {
    for (ll k = 0; k < K; k++) {
        for (ll r = 0; r < R; r++) {
            if (P[t][k][r][n] > 0.0) {
                return true;
            }
        }
    }

    return false;
}

ld BinSearchPower(ll t, ll k, ll r, ll n, ll j, ld availablePower) {
    auto [fid, size, uid, start, duration] = F[j];

    ld p = P[t][k][r][n];
    SetPower(t, k, r, n, j, availablePower);

    ld signal = CalcSignal(t, n);
    if (Completeness[j] + signal < size) {
        SetPower(t, k, r, n, j, p);

        return availablePower;
    }

    ld lx = 0.0;
    ld rx = availablePower;

    while (rx - lx > 0.01) {
        ld m = (lx + rx) / 2;
        SetPower(t, k, r, n, j, m);

        ld signal = CalcSignal(t, n);
        if (Completeness[j] + signal < size) {
            lx = m;
        } else {
            rx = m;
        }
    }

    SetPower(t, k, r, n, j, p);
    return rx;
}

ll ScheduleOneMode = 0;

ll ScheduleOne(ll j, bool simulate = false) {
    auto [fid, size, uid, start, duration] = F[j];
    ll n = uid;
    ld completeness = Completeness[j];

    if (Completeness[j] >= size) {
        return 1;
    }

    vector<tuple<ll, ll, ll, ll>> p;

    set<tuple<ld, ll, ll, ll>, greater<>> o;
    for (ll t = start; t < start + duration; t++) {
        for (ll r = 0; r < R; r++) {
            ld sum = 0.0;

            for (ll k = 0; k < K; k++) {
                sum += S[t][k][r][n];
            }

            o.insert({sum, t, r, n});
        }
    }

    for (auto [_, t, r, n] : o) {
        if (IsRadioAlreadyBooked(t, r, n)) {
            continue;
        }

        set<tuple<ld, ll>, greater<>> q;
        for (ll k = 0; k < K; k++) {
            q.insert({S[t][k][r][n], k});
        }

        for (auto [_, k] : q) {
            ld availableCellPower = CalcCellPowerLeft(t, k);
            ld availableRadioPower = CalcRadioPowerLeft(t, k, r);
            ld availablePower = min(availableCellPower, availableRadioPower);

            if (availablePower < eps) {
                continue;
            }

            Completeness[j] -= CalcSignal(t, n);

            p.push_back({t, k, r, n});
            SetPower(t, k, r, n, j, BinSearchPower(t, k, r, n, j, availablePower));

            ld signal = CalcSignal(t, n);
            Completeness[j] += signal;

            if (Completeness[j] >= size) {
                break;
            }
        }


//        ld completeness = 0.0;
//        for (ll t = start; t < start + duration; t++) {
//            completeness += CalcSignal(t, n);
//        }
//
//        if (abs(completeness - Completeness[j]) > 0.1) {
//            throw 1;
//        }


        if (Completeness[j] >= size) {
            break;
        }
    }

    auto rollback = [&]() {
        Completeness[j] = completeness;

        for (auto [t, k, r, n] : p) {
            SetPower(t, k, r, n, j, 0.0);
        }
    };

    set<ll> rUsed;
    for (auto [t, k, r, n] : p) {
        rUsed.insert(r);
    }

    ll res = rUsed.size();
    if (ScheduleOneMode == 1) {
        res = rUsed.size() * 1000 + p.size();
    }
    if (ScheduleOneMode == 2) {
        res = Rng.GetRandomNumber();
    }
    if (ScheduleOneMode == 3) {
        res = rUsed.size() * 1000 + Rng.GetRandomNumber() % 1000;
    }
    if (ScheduleOneMode == 4) {
        res = p.size() * 1000 + Rng.GetRandomNumber() % 1000;
    }

    if (Completeness[j] < size) {
        rollback();
        return 1e9;
    }

    if (simulate) {
        rollback();
    }

//    if (res > 1) {
//        for (auto [t, k, r, n] : p) {
//            P[t][k][r][n] = 0.0;
//        }
//    }

    return res;
}

ld CalcCompletedFrames() {
    ld res = 0;

    for (auto [fid, size, uid, start, duration] : F) {
//        ld completeness = 0.0;
//        for (ll t = start; t < start + duration; t++) {
//            completeness += CalcSignal(t, uid);
//        }
//        if (abs(completeness - Completeness[fid]) > 0.1) {
////            throw 1;
//        }
        if (size <= Completeness[fid]) {
            res += 1;
        }
    }

    return res;
}

ld CalcConsumedPower() {
    ld res = 0;

    for (ll t = 0; t < T; t++) {
        for (ll k = 0; k < K; k++) {
            for (ll r = 0; r < R; r++) {
                for (ll n = 0; n < N; n++) {
                    res += P[t][k][r][n];
                }
            }
        }
    }

    return res;
}

ld CalcScore() {
    ld completedFrames = CalcCompletedFrames();
    ld consumedPower = CalcConsumedPower();

    return completedFrames - 0.000001 * consumedPower;
}

vector<ll> EstimatedDuration;

ll DummySolve(ll mode) {
    auto Q = F;

    ScheduleOneMode = mode % 5;
    EstimatedDuration = vector<ll>(J);
    for (ll j = 0; j < J; j++) {
        if (CheckTL()) {
            return 0;
        }

        EstimatedDuration[j] = ScheduleOne(j, true);
    }

    sort(Q.begin(), Q.end(),[&](const auto &a, const auto &b) {
        auto get_key = [&](auto& a) {
            auto [fid, size, uid, start, duration] = a;
            return make_tuple(EstimatedDuration[fid], size, duration, fid);
        };
        return get_key(a) < get_key(b);
    });

    ll cnt = 0;
    bool halfBreak = false; // mode % 2;

    for (auto [fid, size, uid, start, duration] : Q) {
        if (CheckTL()) {
            return 0;
        }
        ScheduleOne(fid);
        cnt += 1;
        if (halfBreak && cnt * 2 >= Q.size()) {
            break;
        }
    }

    if (halfBreak) {
        for (ll j = 0; j < J; j++) {
            if (CheckTL()) {
                return 0;
            }

            EstimatedDuration[j] = ScheduleOne(j, true);
        }

        sort(Q.begin(), Q.end(),[&](const auto &a, const auto &b) {
            auto get_key = [&](auto& a) {
                auto [fid, size, uid, start, duration] = a;
                return make_tuple(EstimatedDuration[fid], size, duration, fid);
            };
            return get_key(a) < get_key(b);
        });

        for (auto [fid, size, uid, start, duration] : Q) {
            if (CheckTL()) {
                return 0;
            }
            ScheduleOne(fid);
        }
    }

    for (auto [fid, size, uid, start, duration] : Q) {
        if (CheckTL()) {
            return 0;
        }

        RadioPowerLimit = 4.0;
        ScheduleOne(fid);
        RadioPowerLimit = 1.0;
    }



    return 0;
}

ld InspectDuration() {
    map<ll, ll> d;

    for (ll j = 0; j < J; j++) {
        auto [fid, size, n, start, duration] = F[j];
        d[duration] += 1;
    }

    double res = 0.0;
    for (auto [duration, cnt] : d) {
        cerr << duration << " " << cnt << endl;
        res += (ld)(duration * cnt) / J;
    }

    return res;
}

vector<ll> PrefilterFrames(vector<ll>& frames, ll t, map<ll, set<ll>> cells, map<ll, ld>& avgS) {
    vector<tuple<ld, ld, ll>> o;
    for (auto j : frames) {
        auto [fid, size, n, start, duration] = F[j];

        ld power = 4.0 - Eps;
        for (auto [k, availableRadios] : cells) {
            for (auto r : availableRadios) {
                auto [fid, size, n, start, duration] = F[j];
                SetPower(t, k, r, n, j, power);
            }
        }

        ld signal = CalcSignal(t, n);
        o.push_back({signal / size, size / log2(1.1 + avgS[j]), j});

        for (auto [k, availableRadios] : cells) {
            for (auto r : availableRadios) {
                auto [fid, size, n, start, duration] = F[j];
                SetPower(t, k, r, n, j, 0.0);
            }
        }
    }

    sort(o.begin(), o.end());
    for (auto [_, _1, j] : o) {
        ld power = 4.0 - Eps;

        for (auto [k, availableRadios] : cells) {
            for (auto r : availableRadios) {
                auto [fid, size, n, start, duration] = F[j];
                SetPower(t, k, r, n, j, power);
            }
        }
    }

    vector<ll> filteredFrames;
    set<ll> u;
    for (ll i = o.size() - 1; i >= 0; i--) {
        auto [_, _1, j] = o[i];
        auto [fid, size, n, start, duration] = F[j];

        ld signal = CalcSignal(t, n);
        if (signal >= size) {
            filteredFrames.push_back(j);
            u.insert(j);
        } else {
            for (auto [k, availableRadios] : cells) {
                for (auto r : availableRadios) {
                    SetPower(t, k, r, n, j, 0.0);
                }
            }
        }
    }

    for (auto [_, _1, j] : o) {
        auto [fid, size, n, start, duration] = F[j];

        if (u.count(j)) {
            continue;
        }

        ld power = 4.0 - Eps;

        for (auto [k, availableRadios] : cells) {
            for (auto r : availableRadios) {
                auto [fid, size, n, start, duration] = F[j];
                SetPower(t, k, r, n, j, power);
            }
        }
        ld signal = CalcSignal(t, n);
        if (signal >= size) {
            filteredFrames.push_back(j);
            u.insert(j);
        } else {
            for (auto [k, availableRadios] : cells) {
                for (auto r : availableRadios) {
                    SetPower(t, k, r, n, j, 0.0);
                }
            }
        }
    }

    ll extra = 1;
    for (auto [_, _1, j] : o) {
        if (extra <= 0) {
            break;
        }

        if (u.count(j)) {
            continue;
        }

        filteredFrames.push_back(j);
        u.insert(j);

        extra--;
    }

    for (auto [_, _1, j] : o) {
        for (auto [k, availableRadios] : cells) {
            for (auto r : availableRadios) {
                auto [fid, size, n, start, duration] = F[j];
                SetPower(t, k, r, n, j, 0.0);
            }
        }
    }

//    cerr << frames.size() << "->" << filteredFrames.size() << endl;

    return filteredFrames;
}

ll MultiRadioSolve(vector<ll> frames, ll t, map<ll, set<ll>> cells, ld powerLimit, bool complete, bool improve) {
//    ld availablePower = J > 1000 ? 10000.0 : 10000.0;
//    ld eps = (J > 1000 ? 0.01 : 0.01);
//    if (!improve && J < 1000) {
//        availablePower = 1.0;
//        eps = 0.01;
//    }

    ld availablePower = 1000.0;
    ld eps = 0.01;

    map<ll, map<ll, ld>> usedPowerCellRadio;
    for (auto [k, radios] : cells) {
        for (auto r : radios) {
            for (ll n = 0; n < N; n++) {
                usedPowerCellRadio[k][r] += P[t][k][r][n];
            }
        }
    }

    map<ll, ld> avgS;
    for (auto j : frames) {
        auto [fid, size, n, start, duration] = F[j];

        ll cnt = 0;
        for (auto [k, radios] : cells) {
            for (auto r : radios) {
                avgS[j] += S[t][k][r][n];
                cnt += 1;
            }
        }

        if (cnt > 0) {
            avgS[j] /= cnt;
        }
        if (avgS[j] <= 0) {
            avgS[j] = 1.0;
        }
    }


    auto getMult = [&](set<ll>& availableRadios, ll k, ll r) {
        ld allowedPower = min((ld)4.0 - Eps - usedPowerCellRadio[k][r], min((ld)(R - Eps) / availableRadios.size() - usedPowerCellRadio[k][r], powerLimit / availableRadios.size()));
        ld allowedPowerBase = min((ld)powerLimit, min((ld)4.0 - Eps, (ld)R - Eps));

        ld mult = allowedPowerBase > 0 ? allowedPower / allowedPowerBase : 1.0;
        return mult;
    };

    auto originalFrames = frames;

    map<ll, ld> completeness;
    for (auto j : frames) {
        auto [fid, size, n, start, duration] = F[j];
        completeness[j] = Completeness[j] - CalcSignal(t, n);
    }

    // prefilter
    if (improve) {
        frames = PrefilterFrames(frames, t, cells, avgS);
    }



    auto evaluate = [&](map<ll, set<ll>> cells, vector<ll>& frames) {
        for (auto [k, availableRadios] : cells) {
            for (auto r : availableRadios) {
                for (auto j : frames) {
                    auto [fid, size, n, start, duration] = F[j];

                    ld mult = getMult(availableRadios, k, r);
                    SetPower(t, k, r, n, j, mult * availablePower / frames.size());
                }
            }
        }

        vector<tuple<ld, ll, ll>> q;
        for (auto j : frames) {
            auto [fid, size, n, start, duration] = F[j];

            ld lx = 0.0;
            ld rx = availablePower;

//            ld eps = max(0.01, lx * 0.01);
            ll it = 0;
            while (rx - lx > max(eps, lx * eps) && it < 100) {
                it++;
                ld mx = (lx + rx) / 2;

                for (auto [k, availableRadios] : cells) {
                    for (auto r : availableRadios) {
                        for (auto j : frames) {
                            auto [fid, size, n, start, duration] = F[j];

                            ld mult = getMult(availableRadios, k, r);
                            SetPower(t, k, r, n, j, mult * mx);
                            Tm--;
                        }
                    }
                }


                ld signal = CalcSignal(t, n);
                if (signal + completeness[j] < size) {
                    lx = mx;
                } else {
                    rx = mx;
                }
            }
            Tm++;

            for (auto [k, availableRadios] : cells) {
                for (auto r : availableRadios) {
                    for (auto j : frames) {
                        auto [fid, size, n, start, duration] = F[j];

                        ld mult = getMult(availableRadios, k, r);
                        SetPower(t, k, r, n, j, mult * availablePower / frames.size());
                    }
                }
            }

            q.push_back({rx, (ld)size / log2(1.1 + avgS[j]), j});
        }


        for (auto [k, availableRadios] : cells) {
            for (auto r : availableRadios) {
                for (auto j : frames) {
                    auto [fid, size, n, start, duration] = F[j];

                    SetPower(t, k, r, n, j, 0.0);
                }
            }
        }


        sort(q.begin(), q.end());
        return q;
    };

    auto z = evaluate(cells, frames);

    ld maxSum = min((ld)powerLimit, min((ld)4.0 - Eps, (ld)R - Eps));

    ll minExtra = 4;
    while (improve) {
        ll cnt = 0;
        ld sum = 0.0;
        for (ll i = 0; i < z.size(); i++) {
            auto &[val, size, j] = z[i];
            if (sum + val > maxSum) {
                break;
            }
            sum += val;
            cnt += 1;
        }

        ll extra = max((ll)minExtra, (ll)(z.size() - cnt) / 2);
        minExtra /= 2;
        if (z.size() == cnt) {
            break;
        }
        while (z.size() > cnt + extra) {
            z.pop_back();
        }

        frames = vector<ll>();
        for (auto [_, size, j] : z) {
            frames.push_back(j);
        }

        z = evaluate(cells, frames);

        if (extra <= 1) {
            break;
        }
    }

    ll cnt = 0;
    ld sum = 0.0;
    for (ll i = 0; i < z.size(); i++) {
        auto &[val, size, j] = z[i];
        if (sum + val > maxSum) {
            if (improve && complete) {
                val = maxSum - sum;
                cnt += 1;
            }
            break;
        }
        sum += val;
        cnt += 1;
    }


    for (ll i = 0; i < cnt; i++) {
        auto [val, _, j] = z[i];
        auto [fid, size, n, start, duration] = F[j];

        for (auto [k, availableRadios] : cells) {
            for (auto r: availableRadios) {
                ld mult = getMult(availableRadios, k, r);
                SetPower(t, k, r, n, j, val * mult);
            }
        }
    }

    for (auto j : originalFrames) {
        auto [fid, size, n, start, duration] = F[j];

        Completeness[j] = completeness[j] + CalcSignal(t, n);
    }

    ll res = 0;
    for (ll i = 0; i < cnt; i++) {
        auto [val, _, j] = z[i];
        auto [fid, size, n, start, duration] = F[j];

        if (Completeness[j] >= size) {
            res += 1;
        }
    }

    return res;
}

ll LowDurationSolve(ll t);

ll MultiRadioSolve(ll it) {
    ll res = 0;

    for (ll t = 0; t < T; t++) {
        if (CheckTL()) {
            break;
        }
        auto activeFrames = Frames[t];

        map<ll, set<ll>> cells;

        vector<ll> radios;
        for (ll r = 0; r < R; r++) {
            radios.push_back(r);
        }

        shuffle(radios.begin(), radios.end(), Rng.Rng);

        for (ll i = 0; i < R; i++) {
            cells[(i + it) % K].insert(radios[i]);
        }

        map<ll, set<ll>> smallCell;
        for (auto [k, radios] : cells) {
            if (smallCell.empty() || smallCell.begin()->second.size() < radios.size()) {
                smallCell.clear();
                smallCell[k] = radios;
            }
        }


        ld powerLimit = max(1.0, min(4.0 - Eps, (R - Eps - 4.0)));

        ll c0 = MultiRadioSolve(activeFrames, t, smallCell, powerLimit, false, false);

        vector<ll> completedFrames;
        vector<ll> filteredFrames;
        for (auto j: activeFrames) {
            auto [fid, size, n, start, duration] = F[j];

            if (Completeness[j] >= size) {
                completedFrames.push_back(j);
                continue;
            }

            for (auto [k, radios]: smallCell) {
                for (auto r: radios) {
                    SetPower(t, k, r, n, j, 0.0);
                }
            }

            Completeness[j] = 0.0;
            filteredFrames.push_back(j);
        }

        ll c1 = MultiRadioSolve(filteredFrames, t, cells, 1e6, true, true);

        for (auto j : completedFrames) {
            auto [fid, size, n, start, duration] = F[j];

            Completeness[j] = CalcSignal(t, n);
        }
        res = c0 + c1;

        if (BestRes[t] < res) {
            BestRes[t] = res;
            BestP[t] = P[t];
            BestRadioUsage[t] = RadioUsage[t];

            for (auto j : Frames[t]) {
                BestCompleteness[j] = Completeness[j];
            }
        }

        for (auto j : activeFrames) {
            auto [fid, size, n, start, duration] = F[j];

            for (ll k = 0; k < K; k++) {
                for (ll r = 0; r < R; r++) {
                    SetPower(t, k, r, n, j, 0.0);
                }
            }

            Completeness[j] = 0;
        }
    }

    return res;
}

ll LowDurationSolve(ll t, ll it) {
    set<ll> usedCells = {};
    set<ll> usedRadios = {};
    ll resSum = 0.0;

    vector<ll> radios;
    for (ll r = 0; r < R; r++) {
        radios.push_back(r);
    }
    shuffle(radios.begin(), radios.end(), Rng.Rng);


    for (auto r : radios) {
        ll bc = 0;
        ll bk = -1;



        vector<ll> availableFrames;
        for (auto j : Frames[t]) {
            auto [fid, size, n, start, duration] = F[j];
            if (Completeness[j] >= size) {
                continue;
            }

            availableFrames.push_back(j);
        }

        if (r != radios.front()) {
            for (ll k = 0; k < K; k++) {
                if (usedCells.count(k)) {
                    continue;
                }

                if (CheckTL()) {
                    return 0;
                }

                map<ll, set<ll>> cells;
                cells[k] = {r};

                ll res = MultiRadioSolve(availableFrames, t, cells, 1e6, false, false);
                if (bc < res) {
                    bc = res;
                    bk = k;
                }

                for (auto j : availableFrames) {
                    auto [fid, size, n, start, duration] = F[j];

                    SetPower(t, k, r, n, j, 0.0);
                    Completeness[j] = CalcSignal(t, n);
                }
            }
        } else {
            vector<ll> qCells;
            for (ll k = 0; k < K; k++) {
                if (usedCells.count(k)) {
                    continue;
                }
                qCells.push_back(k);
            }
            if (qCells.size()) {
                bk = qCells[Rng.GetRandomNumber() % qCells.size()];
            }
        }



        ll res = 0.0;
        if (bk >= 0) {
            map<ll, set<ll>> cells;
            cells[bk] = {r};

            res = MultiRadioSolve(availableFrames, t, cells, 1e6,false, true);
            resSum += res;
            usedCells.insert(bk);
            usedRadios.insert(r);
        } else {
            break;
        }
    }

    if (usedCells.size() < K) {
        vector<tuple<ld, ll, ll>> o;

        vector<ld> cellPower(K);
        for (ll k = 0; k < K; k++) {
            for (ll r = 0; r < R; r++) {
                for (ll n = 0; n < N; n++) {
                    cellPower[k] += P[t][k][r][n];
                }
            }
        }
        ld defaultAvailablePower = min((ld)4.0 - Eps, (ld)R - Eps);

        for (ll r = 0; r < R; r++) {
            if (usedRadios.count(r)) {
                continue;
            }

            for (auto j: Frames[t]) {
                auto [fid, size, n, start, duration] = F[j];
                if (Completeness[j] >= size) {
                    continue;
                }

                ld lx = 0.0;
                ld rx = defaultAvailablePower;

                while (rx - lx > 0.01) {
                    ld mx = (lx + rx) / 2;

                    for (ll k = 0; k < K; k++) {
                        ld availablePower = min((ld)(4.0 - Eps - cellPower[k]), (ld)R - Eps);
                        ld mult = availablePower / defaultAvailablePower;

                        SetPower(t, k, r, n, j, mx * mult); // R / usedRadios.size());
                    }

                    ld signal = CalcSignal(t, n);
                    if (signal > size) {
                        rx = mx;
                    } else {
                        lx = mx;
                    }

                    for (ll k = 0; k < K; k++) {
                        SetPower(t, k, r, n, j, 0.0);
                    }
                }

                o.push_back({rx, r, j});
            }
        }

        sort(o.begin(), o.end());

        ld sum = 0.0;
        ll cnt = 0;
        ll ur = usedRadios.size();

        for (auto [val, r, j]: o) {
            if (usedRadios.count(r)) {
                continue;
            }

            auto [fid, size, n, start, duration] = F[j];
            if (Completeness[j] >= size) {
                continue;
            }

            if (sum + val > R) {
                continue;
            }

            for (ll k = 0; k < K; k++) {
                ld availablePower = min((ld)(4.0 - Eps - cellPower[k]), (ld)R - Eps);
                ld mult = availablePower / defaultAvailablePower;

                SetPower(t, k, r, n, j, mult * val);
            }

            Completeness[j] = CalcSignal(t, n);
            sum += val;
            cnt += 1;
            usedRadios.insert(r);
        }

        for (auto j : Frames[t]) {
            auto [fid, size, n, start, duration] = F[j];

            if (Completeness[j] >= size) {
                continue;
            }

            set<ll> availableRadios;
            for (ll r = 0; r < R; r++) {
                if (usedRadios.count(r)) {
                    continue;
                }
                availableRadios.insert(r);
            }

            vector<ld> cellPower(K);
            for (ll k = 0; k < K; k++) {
                for (ll r = 0; r < R; r++) {
                    for (ll n = 0; n < N; n++) {
                        cellPower[k] += P[t][k][r][n];
                    }
                }
            }

            for (ll k = 0; k < K; k++) {
                for (auto r : availableRadios) {
                    SetPower(t, k, r, n, j, min((ld)4.0 - Eps, (R - Eps - cellPower[k])) / availableRadios.size());
                }
            }

            Completeness[j] = CalcSignal(t, n);
            if (Completeness[j] >= size) {
                break;
            }

            for (ll k = 0; k < K; k++) {
                for (auto r : availableRadios) {
                    SetPower(t, k, r, n, j, 0.0);
                }
            }

            Completeness[j] = CalcSignal(t, n);
        }
    }

    ll res = 0;
    for (auto j : Frames[t]) {
        auto [fid, size, n, start, duration] = F[j];
        if (Completeness[j] < size) {
            continue;
        }
        res += 1;
    }

    return res;
}


ll LowDurationSolve(ll it) {
    for (ll t = 0; t < T; t++) {
        if (CheckTL()) {
            break;
        }

        ll res = LowDurationSolve(t, it);

        if (BestRes[t] < res) {
            BestRes[t] = res;
            BestP[t] = P[t];
            BestRadioUsage[t] = RadioUsage[t];

            for (auto j : Frames[t]) {
                BestCompleteness[j] = Completeness[j];
            }
        }

        for (auto j : Frames[t]) {
            auto [fid, size, n, start, duration] = F[j];

            for (ll k = 0; k < K; k++) {
                for (ll r = 0; r < R; r++) {
                    SetPower(t, k, r, n, j, 0.0);
                }
            }

            Completeness[j] = 0;
        }
    }

    return 0;
}

bool IsOneshot(ll t, ll j) {
    auto [fid, size, n, start, duration] = F[j];

    ll br = 0;
    ll bk = 0;

    for (ll r = 0; r < R; r++) {
        for (ll k = 0; k < K; k++) {
            if (S[t][bk][br][n] < S[t][k][r][n]) {
                bk = k;
                br = r;
            }
        }
    }

    SetPower(t, bk, br, n, j, 4.0 - Eps);
    ld signal = CalcSignal(t, n);
    SetPower(t, bk, br, n, j, 0.0);

    return signal > size;
}

ll DefaultSolve(ll mode) {
    set<ll> activeFrames;
    set<ll> oneshotFrames;

    for (ll t = 0; t < T; t += 1) {
        if (CheckTL()) {
            return 0;
        }

        for (auto j : Frames[t]) {
            if (!activeFrames.count(j)) {
                if (IsOneshot(t, j)) {
                    oneshotFrames.insert(j);
                }
                activeFrames.insert(j);
            }
        }

        ll ad = (ll)round(AvgDuration);
        if (ad == 0) {
            ad = 1;
        }
        if ((t + 1) % ad == 0) {
            ll x = Rng.GetRandomNumber() % K;

            vector<ll> radios;
            for (ll r = 0; r < R; r++) {
                radios.push_back(r);
            }
            shuffle(radios.begin(), radios.end(), Rng.Rng);

            set<ll> usedCells;
            for (auto r : radios) {
                ll bk = -1;
                ll bc = 0;

                vector<ll> frames;
                for (auto j : oneshotFrames) {
                    frames.push_back(j);
                }

                if (mode % 2) {
                    if (r == radios.front()) {
                        bk = Rng.GetRandomNumber() % K;
                    } else {
                        for (ll k = 0; k < K; k++) {
                            if (usedCells.count(k)) {
                                continue;
                            }

                            if (CheckTL()) {
                                return 0;
                            }

                            map<ll, set<ll>> cells;
                            cells[k] = {r};

                            ll res = MultiRadioSolve(frames, t, cells, 1e6, false, false);
                            if (bc < res) {
                                bc = res;
                                bk = k;
                            }

                            for (auto j : frames) {
                                auto [fid, size, n, start, duration] = F[j];

                                SetPower(t, k, r, n, j, 0.0);
                                Completeness[j] = CalcSignal(t, n);
                            }
                        }
                    }
                } else {
                    bk = (r + x) % K;
                }

                if (bk < 0) {
                    break;
                }

                map<ll, set<ll>> cells;
                cells[bk].insert(r);
                usedCells.insert(bk);

                ld powerLimit = CalcCellPowerLeft(t, bk);
                MultiRadioSolve(frames, t, cells, powerLimit, false, true);

                for (auto j : frames) {
                    auto [fid, size, n, start, duration] = F[j];

                    if (Completeness[j] < size) {
                        continue;
                    }

                    oneshotFrames.erase(j);
                }
            }
        }

        for (auto j : Frames[t]) {
            auto [fid, size, n, start, duration] = F[j];

            if (t + 1 == start + duration) {
                activeFrames.erase(j);
                oneshotFrames.erase(j);
            }
        }
    }

    DummySolve(mode);

    ll res = CalcCompletedFrames();
    if (Res < res) {
        Res = res;
        BestP = P;
        BestCompleteness = Completeness;
        BestRadioUsage = RadioUsage;
    }

    Reset();

    return res;
}

ll DummySolve() {
    DummySolve(0);

    ll res = CalcCompletedFrames();
    if (Res < res) {
        Res = res;
        BestP = P;
        BestCompleteness = Completeness;
        BestRadioUsage = RadioUsage;
    }

    Reset();

    return 0;
}

ld InspectD() {
    ld sum = 0.0;
    ll cnt = 0;

    for (ll k = 0; k < K; k++) {
        for (ll r = 0; r < R; r++) {
            for (ll n = 0; n < N; n++) {
                for (ll m = n + 1; m < N; m++) {
                    sum += D[k][r][n][m];
                    cnt += 1;
                }
            }
        }
    }

    return sum / cnt;
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); cout.precision(15); cout.setf(ios::fixed); cerr.precision(6); cerr.setf(ios::fixed);
    StartTm = std::chrono::steady_clock::now();


    int mode = argc - 1;

    if (mode == 0) {
        ReadInput(cin);
    } if (mode == 1) {
        ifstream in(argv[1]);
        ReadInput(in);
    } else if (mode == 2) {
        // GenerateInputLowDuration(100, 5, 10, 100, 500);
        GenerateInputLowDuration(1000, 10, 10, 100, 5000);
    }

    cerr << T << " " << K << " " << R << " " << N << " " << J << endl;

    cerr << "avgDuration = " << AvgDuration << endl;
    cerr << "avgD = " << AvgD << endl;

    ll res = 0;
    if (false) {
        for (ll it = 0; it < 1000 && !CheckTL(); it++) {
            MultiRadioSolve(it + 2);
        }
    } else if (N == 2 && R == 1 && K == 2) {
        res = DummySolve();
    } else if (AvgDuration < 1.5) {
        DummySolve();
        LowDurationSolve(0);

        if (AvgD < 0.25) {
            for (ll it = 0; it < 1000 && !CheckTL(); it++) {
                MultiRadioSolve(it + 2);
            }
        } else {
            for (ll it = 1; it < 1000 && !CheckTL(); it++) {
                LowDurationSolve(it);
            }
        }
    } else {
        DummySolve();

        for (ll it = 0; it < 1000 && !CheckTL(); it++) {
            res = DefaultSolve(it);
        }
    }


    P = BestP;
    RadioUsage = BestRadioUsage;
    Completeness = BestCompleteness;

    CheckPowerInvariant();
    WriteOutput();


    ld expectedScore = CalcScore();
    cerr << "expectedScore = " << expectedScore << endl;

//    AllocateMemory(100 * expectedScore / J);

    auto stop = std::chrono::steady_clock::now();
    cerr << "Time = " << duration_cast<milliseconds>((stop - StartTm)).count() << endl;
    return 0;
}

