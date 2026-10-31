import solvers.«88.py.lean.d».Loop

namespace solvers.«88».dfs
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.ProductSumNumbers
set_option Elab.async false

abbrev parameters : List Arg := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).1
def arguments (start : Nat) (before : List Nat) (context : Context) : List Mutable.Value :=
  [number start, number before.prod, number before.sum, number before.length,
    .ref context.bestCell, .ref context.maxCell, .ref context.limitCell]
def initialEnv (start : Nat) (before : List Nat) (context : Context) : HeapCalls.Env :=
  functionEnv parameters dfsCode (arguments start before context) []

/-- The complete recursive closure terminates and computes the exact shared-table update. -/
theorem result (maxSize limit start : Nat) (before : List Nat) (valid : Admissible before)
    (lower : 2 ≤ start) (context : Context) (cells : Cells context maxSize limit)
    (best : List Nat) (length : best.length = maxSize + 1) :
    SearchReturns maxSize limit start before context best := by
  have descendants : Descendants maxSize limit before context := by
    intro factor factorBound inside best length
    have single : Admissible [factor] := by simpa [Admissible] using factorBound
    exact result maxSize limit factor (before ++ [factor])
      (admissible_append before [factor] valid single) factorBound context cells best length
  have positive := product_positive before valid
  let initial := initialEnv start before context
  let measured := setLocal initial "max_f" (number (limit / before.prod))
  have divisionValue (a b : Nat) (positive : 0 < b) :
      Mutable.binary .floorDiv (number a) (number b) (working context best) =
        some (.ok (number (a / b)), working context best) := by
    simp [Mutable.binary, scalarBinary, number, Int.natCast_ediv, Nat.ne_of_gt positive, positive]
  have division : BlockContinues program initial [dfsCode[0]'(by decide)] (working context best)
      measured (working context best) := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := number limit) (bv := number before.prod)
      (captured_number (working context best) context.limitCell limit "__formalsw_cell_solve_limit"
        initial rfl (cells.bound best)) (Evaluates.lookup rfl)
    exact divisionValue limit before.prod (by omega)
  have ready : Locals measured context before := by constructor <;> rfl
  obtain ⟨finalEnv, executed, _⟩ := loop maxSize limit before valid start lower context cells descendants
    best length measured ready
  have ranged : BlockContinues program measured [dfsCode[1]'(by decide)] (working context best) finalEnv
      (working context (accumulate (search maxSize limit start before.prod before.sum before.length) best)) := by
    apply BlockContinues.for_range (values := [number start, number (limit / before.prod + 1)])
      (middle := working context best)
    · exact .cons (Evaluates.lookup rfl)
        (.cons (Evaluates.binary (av := number (limit / before.prod)) (bv := number 1)
          (Evaluates.lookup rfl) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])) .nil)
    · rfl
    · exact executed
  apply BlockReturns.function rfl rfl
  apply division.returns
  apply ranged.returns
  exact fun _ _ => .next rfl .refl
termination_by limit + 1 - before.prod
decreasing_by
  simp only [List.prod_append, List.prod_cons, List.prod_nil, mul_one]
  exact descent limit before.prod factor (product_positive before valid) factorBound inside

theorem root (maxSize : Nat) (context : Context) (cells : Cells context maxSize (2 * maxSize)) :
    ReturnsValue program "__formalsw_local_solve_dfs"
      [number 2, number 1, number 0, number 0, .ref context.bestCell, .ref context.maxCell, .ref context.limitCell]
      (working context (List.replicate (maxSize + 1) (2 * maxSize + 1))) (.scalar .none)
      (working context (table maxSize)) := by
  exact result maxSize (2 * maxSize) 2 [] (by simp [Admissible]) (by omega) context cells
    (List.replicate (maxSize + 1) (2 * maxSize + 1)) (by simp)

end solvers.«88».dfs
