import «.formalsw».solvers.«19.py.ast»
import lang.py.FormalSW.HeapRuntimeCostMembership
import lang.py.FormalSW.MembershipCalls

namespace solvers.«19».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
private abbrev program := Solvers.P19Ast.module

private theorem remainder_test (year divisor : Int) (positive : 0 < divisor)
    (heap : Heap) (env : HeapCalls.Env) (lookup : env "year" = some (.scalar (.int year))) :
    EvaluatesIn .word program env
      (.compare (.binOp (.name "year") .mod (.constant (.int divisor))) [(.eq, .constant (.int 0))])
      heap 12 (.scalar (.bool (decide (year % divisor = 0)))) heap := by
  apply EvaluatesIn.compare (ca := 6) (cb := 1) (work := 2)
    (av := .scalar (.int (year % divisor))) (bv := .scalar (.int 0))
    (EvaluatesIn.binary (op := .mod) (work := 1) (EvaluatesIn.lookup lookup) EvaluatesIn.constant
      (by simp [Mutable.binary, scalarBinary, show divisor ≠ 0 by omega, positive]) rfl) EvaluatesIn.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  · rfl

theorem leap_cost (year : Int) (heap : Heap) :
    ∃ flag : Bool, ReturnsIn .word program "is_leap_year" [.scalar (.int year)] heap
      50 (.scalar (.bool flag)) heap := by
  let args := ((findFunctionIn program.body "is_leap_year").getD ([], [])).1
  let body := ((findFunctionIn program.body "is_leap_year").getD ([], [])).2
  let env := functionEnv args body [.scalar (.int year)] []
  have fourHundred := remainder_test year 400 (by decide) heap env rfl
  have hundred := remainder_test year 100 (by decide) heap env rfl
  have fourth := remainder_test year 4 (by decide) heap env rfl
  suffices ∃ flag : Bool, BlockReturnsIn .word program env (executableBody body) heap 47
      (.scalar (.bool flag)) heap by
    obtain ⟨flag, returned⟩ := this
    refine ⟨flag, ?_⟩
    exact BlockReturnsIn.function (module := program) (name := "is_leap_year")
      (args := args) (body := body) (values := [.scalar (.int year)]) rfl rfl returned
  by_cases century : year % 400 = 0
  · refine ⟨true, ?_⟩
    apply BlockReturnsIn.mono (small := 16) (larger := by omega)
    apply BlockReturnsIn.branch (cc := 12) (cb := 2) (flag := true) (value := .bool true)
      (by simpa only [century, decide_true] using fourHundred) rfl
    exact BlockReturnsIn.return_value EvaluatesIn.constant
  · by_cases hundredYear : year % 100 = 0
    · refine ⟨false, ?_⟩
      apply BlockReturnsIn.mono (small := 30) (larger := by omega)
      apply BlockReturnsIn.branch (cc := 12) (cb := 16) (flag := false) (value := .bool false)
        (by simpa only [century, decide_false] using fourHundred) rfl
      apply BlockReturnsIn.branch (cc := 12) (cb := 2) (flag := true) (value := .bool true)
        (by simpa only [hundredYear, decide_true] using hundred) rfl
      exact BlockReturnsIn.return_value EvaluatesIn.constant
    · refine ⟨decide (year % 4 = 0), ?_⟩
      apply BlockReturnsIn.mono (small := 41) (larger := by omega)
      apply BlockReturnsIn.branch (cc := 12) (cb := 27) (flag := false) (value := .bool false)
        (by simpa only [century, decide_false] using fourHundred) rfl
      apply BlockReturnsIn.branch (cc := 12) (cb := 13) (flag := false) (value := .bool false)
        (by simpa only [hundredYear, decide_false] using hundred) rfl
      exact BlockReturnsIn.return_value fourth

private abbrev shortMonths : List Mutable.Value := ([4, 6, 9, 11] : List Int).map (fun n => .scalar (.int n))

private theorem short_months_cost (month : Int) (heap : Heap) (env : HeapCalls.Env)
    (lookup : env "month" = some (.scalar (.int month))) :
    EvaluatesIn .word program env (.membership (.name "month")
      (.tuple [.constant (.int 4), .constant (.int 6), .constant (.int 9), .constant (.int 11)]) false)
      heap 40 (.scalar (.bool (decide (month ∈ ([4, 6, 9, 11] : List Int))))) (heap ++ [.tuple shortMonths]) := by
  apply EvaluatesIn.membership (budget := 25) (work := 13)
    (value := .scalar (.int month)) (receiver := .ref heap.length)
  · exact ArgumentsIn.cons (EvaluatesIn.lookup lookup)
      (ArgumentsIn.cons (EvaluatesIn.tuple (ArgumentsIn.cons EvaluatesIn.constant
        (ArgumentsIn.cons EvaluatesIn.constant (ArgumentsIn.cons EvaluatesIn.constant
          (ArgumentsIn.cons EvaluatesIn.constant ArgumentsIn.nil)))) (by simp)) ArgumentsIn.nil)
  · exact membership_tuple_ints _ heap.length month [4, 6, 9, 11] (by simp) false
  · have stored : (heap ++ [.tuple shortMonths])[heap.length]? = some (.tuple shortMonths) := by simp
    simp only [operation, stored]
    exact sequenceMembership_integers month [4, 6, 9, 11]

/-- The calendar helper charges leap tests or the allocated four-month tuple and its scan. -/
theorem month_cost (year month : Int) (heap : Heap) :
    ∃ extra : Heap, ∃ days : Int, ReturnsIn .word program "days_in_month"
      [.scalar (.int year), .scalar (.int month)] heap 80 (.scalar (.int days)) (heap ++ extra) := by
  let args := ((findFunctionIn program.body "days_in_month").getD ([], [])).1
  let body := ((findFunctionIn program.body "days_in_month").getD ([], [])).2
  let env := functionEnv args body [.scalar (.int year), .scalar (.int month)] []
  have february : EvaluatesIn .word program env (.compare (.name "month") [(.eq, .constant (.int 2))])
      heap 7 (.scalar (.bool (decide (month = 2)))) heap := by
    apply EvaluatesIn.compare (work := 2) (av := .scalar (.int month)) (bv := .scalar (.int 2))
      (EvaluatesIn.lookup rfl) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  by_cases second : month = 2
  · obtain ⟨flag, leap⟩ := leap_cost year heap
    have leapCall := EvaluatesIn.call (ArgumentsIn.cons (EvaluatesIn.lookup (env := env) (name := "year") rfl) ArgumentsIn.nil) leap
    have selected : EvaluatesIn .word program env
        (.ifExp (.call (.name "is_leap_year") [.name "year"]) (.constant (.int 29)) (.constant (.int 28)))
        heap 56 (.scalar (.int (if flag then 29 else 28))) heap := by
      apply EvaluatesIn.conditional (ct := 53) (cb := 1) (value := .bool flag) (flag := flag) leapCall
      · cases flag <;> rfl
      · cases flag <;> exact EvaluatesIn.constant
    have returned := BlockReturnsIn.branch (following := (executableBody body).drop 1) (cc := 7) (cb := 57)
      (flag := true) (value := .bool true) (yes := [.return _]) (no := [])
      (by simpa only [second, decide_true] using february) rfl
      (BlockReturnsIn.return_value (following := (executableBody body).drop 1) selected)
    have called := BlockReturnsIn.function (module := program) (name := "days_in_month")
      (args := args) (body := body) (values := [.scalar (.int year), .scalar (.int month)]) rfl rfl returned
    refine ⟨[], if flag then 29 else 28, ?_⟩
    intro anchors stack
    simpa only [List.append_nil] using (called anchors stack).mono (large := 80) (by change 2 + 2 + 1 + _ ≤ 80; omega)
  · have scan := short_months_cost month heap env rfl
    let days := if month ∈ ([4, 6, 9, 11] : List Int) then 30 else 31
    have selected : BlockReturnsIn .word program env ((executableBody body).drop 1)
        heap 44 (.scalar (.int days)) (heap ++ [.tuple shortMonths]) := by
      by_cases short : month ∈ ([4, 6, 9, 11] : List Int)
      · apply BlockReturnsIn.branch (cc := 40) (cb := 2) (flag := true) (value := .bool true)
          (by simpa only [short, decide_true] using scan) rfl
        simpa only [days, if_pos short] using
          (BlockReturnsIn.return_value (following := [.return (.constant (.int 31))])
            (EvaluatesIn.constant (value := .int 30)))
      · apply BlockReturnsIn.branch (cc := 40) (cb := 2) (flag := false) (value := .bool false)
          (by simpa only [short, decide_false] using scan) rfl
        simpa only [days, if_neg short] using
          (BlockReturnsIn.return_value (following := []) (EvaluatesIn.constant (value := .int 31)))
    have returned : BlockReturnsIn .word program env (executableBody body) heap 53
        (.scalar (.int days)) (heap ++ [.tuple shortMonths]) := by
      apply BlockReturnsIn.branch (cc := 7) (cb := 44) (flag := false) (value := .bool false)
        (by simpa only [second, decide_false] using february) rfl
      exact selected
    have called := BlockReturnsIn.function (module := program) (name := "days_in_month")
      (args := args) (body := body) (values := [.scalar (.int year), .scalar (.int month)]) rfl rfl returned
    refine ⟨[.tuple shortMonths], days, ?_⟩
    intro anchors stack
    exact (called anchors stack).mono (large := 80) (by change 2 + 2 + 1 + _ ≤ 80; omega)

end solvers.«19».runtimeCost
