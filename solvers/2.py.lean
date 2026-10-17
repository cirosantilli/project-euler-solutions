import «.formalsw».solvers.«2.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p2.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P2

namespace solvers.«2»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions.P2

private abbrev program := Solvers.P2Ast.module
private def loop : List Stmt :=
  (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 2

private theorem loop_result (limit a b total : Nat) (ha : 0 < a) (hab : a < b)
    (env : ScalarEnv) (hl : env "limit" = some (.int limit))
    (hea : env "a" = some (.int a)) (heb : env "b" = some (.int b))
    (ht : env "total" = some (.int total)) :
    BlockReturns program env loop (.int (go limit a b total ha hab)) := by
  induction h : limit + 1 - a using Nat.strong_induction_on generalizing a b total env with
  | h measure ih =>
      rw [go.eq_1]
      by_cases hle : a ≤ limit
      · simp only [hle, ↓reduceIte]
        have next (t : Nat) (env' : ScalarEnv)
            (hl' : env' "limit" = some (.int limit)) (ha' : env' "a" = some (.int a))
            (hb' : env' "b" = some (.int b)) (ht' : env' "total" = some (.int t)) :
            BlockReturns program env'
              (.assign [.tuple [.name "a", .name "b"]]
                (.tuple [.name "b", .binOp (.name "a") .add (.name "b")]) :: loop)
              (.int (go limit b (a + b) t (by omega) (by omega))) := by
          apply BlockReturns.unpack (names := ["a", "b"])
            (results := [.int b, .int (a + b)]) rfl rfl
          · exact .cons (Evaluates.lookup hb') (.cons
              (Evaluates.binary (Evaluates.lookup ha') (Evaluates.lookup hb')
                (by simp [scalarBinary])) .nil)
          · apply ih (limit + 1 - b) (by omega) b (a + b) t
            all_goals simp [assignNames, ScalarEnv.set, hl', ht']
        apply BlockReturns.while_loop (value := .bool true)
        · exact Evaluates.compare (Evaluates.lookup hea) (Evaluates.lookup hl)
            (by simp [scalarCompare, evalCmpOp, hle])
        · have parity : Evaluates program env
              (.compare (.binOp (.name "a") .mod (.constant (.int 2)))
                [(.eq, .constant (.int 0))]) (.bool (decide (a % 2 = 0))) := by
            apply Evaluates.compare (av := .int ↑(a % 2)) (bv := .int 0)
            · exact Evaluates.binary (Evaluates.lookup hea) Evaluates.constant (by simp [scalarBinary])
            · exact Evaluates.constant
            · simp [scalarCompare, evalCmpOp, Int.ofNat_dvd_natCast, Nat.dvd_iff_mod_eq_zero]
          by_cases heven : a % 2 = 0
          · simp only [heven, ↓reduceIte]
            apply BlockReturns.branch (value := .bool true)
            · simpa [heven] using parity
            · apply BlockReturns.augAssign (value := .int ↑(total + a))
              · exact Evaluates.binary (Evaluates.lookup ht) (Evaluates.lookup hea)
                  (by simp [scalarBinary])
              · apply next
                all_goals simp [ScalarEnv.set, hl, hea, heb]
          · simp only [heven, ↓reduceIte]
            apply BlockReturns.branch (value := .bool false)
            · simpa [heven] using parity
            · exact next total env hl hea heb ht
      · simp only [hle, ↓reduceIte]
        apply BlockReturns.while_loop (value := .bool false)
        · exact Evaluates.compare (Evaluates.lookup hea) (Evaluates.lookup hl)
            (by simp [scalarCompare, evalCmpOp, hle])
        · exact BlockReturns.return_value (Evaluates.lookup ht)

private theorem solve_result (n : Nat) :
    ReturnsValue program "solve" [.int n] (.int (solve n)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.unpack (names := ["a", "b"]) (results := [.int 1, .int 2]) rfl rfl
  · exact .cons Evaluates.constant (.cons Evaluates.constant .nil)
  · apply BlockReturns.assign Evaluates.constant
    exact loop_result n 1 2 0 (by decide) (by decide) _ rfl rfl rfl rfl

theorem solve.«Equiv(naive)» (n : Int) (hp : Calls.Condition program
      (fun key => if key = "limit" then some (.int n) else none) Solvers.P2Ast.requires_solve_0) :
    (Calls.function program "solve" [.int n]).Equiv
      (Calls.function P2Ast.module "naive" [.int n]) := by
  have hn : n > 1 := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare hp rfl
  lift n to Nat using (by omega : 0 ≤ n)
  have hn : 1 ≤ n := by omega
  have initial : ProjectEulerStatements.P2.naive n =
      ProjectEulerStatements.P2.naive.go n 1 0 := by
    rw [ProjectEulerStatements.P2.naive, ProjectEulerStatements.P2.naive.go.eq_1]
    simp [ProjectEulerStatements.P2.fib, hn]
  have naive_result := data.«project-euler-statements».data.py.project_euler_statements.«p2».naive.result n
  rw [← initial, ProjectEulerSolutions.P2.equiv] at naive_result
  exact (solve_result n).equiv naive_result

end solvers.«2»
