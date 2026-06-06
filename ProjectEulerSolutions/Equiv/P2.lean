import ProjectEulerSolutions.P2
namespace ProjectEulerSolutions.P2

lemma fib_pos (i : Nat) : 0 < ProjectEulerStatements.P2.fib i := by
  have := ProjectEulerStatements.P2.fib_ge_succ i
  exact Nat.lt_of_lt_of_le (Nat.succ_pos i) this

lemma fib_add_two (i : Nat) :
    ProjectEulerStatements.P2.fib (i + 2) =
      ProjectEulerStatements.P2.fib i + ProjectEulerStatements.P2.fib (i + 1) := by
  simp [ProjectEulerStatements.P2.fib]

lemma fib_lt_succ (i : Nat) :
    ProjectEulerStatements.P2.fib i < ProjectEulerStatements.P2.fib (i + 1) := by
  cases i with
  | zero =>
      change 1 < 2
      exact Nat.lt.base 1
  | succ i =>
      change ProjectEulerStatements.P2.fib (i + 1) < ProjectEulerStatements.P2.fib (i + 2)
      rw [fib_add_two i]
      calc
        ProjectEulerStatements.P2.fib (i + 1) = 0 + ProjectEulerStatements.P2.fib (i + 1) := by
          simp
        _ < ProjectEulerStatements.P2.fib i + ProjectEulerStatements.P2.fib (i + 1) := by
          exact Nat.add_lt_add_right (fib_pos i) _

lemma go_fib_eq_naive_go (n i total : Nat) :
    go n (ProjectEulerStatements.P2.fib i) (ProjectEulerStatements.P2.fib (i + 1))
        total (fib_pos i) (fib_lt_succ i) =
      ProjectEulerStatements.P2.naive.go n i total := by
  induction h : n + 1 - ProjectEulerStatements.P2.fib i
      using Nat.strong_induction_on generalizing i total with
  | h m ih =>
      rw [go.eq_1, ProjectEulerStatements.P2.naive.go.eq_1]
      by_cases hle : ProjectEulerStatements.P2.fib i ≤ n
      · simp [hle]
        have hm : n + 1 - ProjectEulerStatements.P2.fib (i + 1) < m := by
          rw [← h]
          have hlt := fib_lt_succ i
          omega
        have hrec :=
          ih (n + 1 - ProjectEulerStatements.P2.fib (i + 1)) hm (i + 1)
            (if ProjectEulerStatements.P2.fib i % 2 = 0 then
              total + ProjectEulerStatements.P2.fib i
            else
              total) rfl
        convert hrec using 1
      · simp [hle]

theorem equiv (n : Nat) : ProjectEulerStatements.P2.naive n = solve n := by
  unfold ProjectEulerStatements.P2.naive solve
  simpa [ProjectEulerStatements.P2.fib] using (go_fib_eq_naive_go n 0 0).symm
end ProjectEulerSolutions.P2
