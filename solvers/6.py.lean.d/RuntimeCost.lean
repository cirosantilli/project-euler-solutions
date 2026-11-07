import «.formalsw».solvers.«6.py.ast»
import lang.py.FormalSW.CallsRuntimeCost
import lang.py.FormalSW.PrimitiveWorkBounds
import Mathlib.Tactic.Linarith

namespace solvers.«6».runtimeCost
open FormalSW.PyAst
open FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 1000000

theorem source_bound (n : Nat) :
    Calls.Runtime.TimeBound .bits Solvers.P6Ast.module "solve" [.int n]
      (4096 * (Nat.log2 (n+1)+1)^2) := by
  apply Exists.intro
  refine ⟨.ok (.int (((n : Int) * (n + 1) / 2) * (n * (n + 1) / 2) -
    n * (n + 1) * (2 * n + 1) / 6)), ?_, ?_⟩
  · repeat first
      | exact RuntimeCost.Finishes.done rfl rfl
      | apply RuntimeCost.Finishes.next <;> first | rfl | skip
  · let N := n + 1
    let L := Nat.log2 (n+1) + 1
    have one : 1 ≤ N := by dsimp [N]; omega
    have logPositive : 1 ≤ L := by dsimp [L]; omega
    have square : N ≤ N^2 := by nlinarith
    have fourth : N^2 ≤ N^4 := by nlinarith [Nat.mul_self_le_mul_self square]
    have oneFourth : 1 ≤ N^4 := by omega
    have product : n*N ≤ N^2 := by dsimp [N]; nlinarith
    have half : n*N/2 ≤ N^2 := Nat.le_trans (Nat.div_le_self _ _) product
    have halfSquared : (n*N/2)*(n*N/2) ≤ N^4 := by
      nlinarith [Nat.mul_self_le_mul_self half]
    have twice : 2*n+1 ≤ 2*N := by dsimp [N]; omega
    have triple : n*N*(2*n+1) ≤ 2*N^4 := by
      have multiplied := Nat.mul_le_mul product twice
      have enlarged := Nat.mul_le_mul_left (N^2) square
      nlinarith
    have divided : n*N*(2*n+1)/6 ≤ 2*N^4 := Nat.le_trans (Nat.div_le_self _ _) triple
    have width (value : Nat) (bounded : value ≤ 8*N^4) :
        PrimitiveWork.bitLength (value : Int) ≤ 8*L := by
      have estimate := PrimitiveWork.bitLength_polynomial n value 4 3 bounded
      dsimp [L]
      omega
    have wN := width n (by dsimp [N] at *; omega)
    have wNext := width (n+1) (by change N ≤ 8*N^4; omega)
    have wProduct := width (n*(n+1)) (by change n*N ≤ 8*N^4; omega)
    have wHalf := width (n*(n+1)/2) (by change n*N/2 ≤ 8*N^4; omega)
    have wSquare := width ((n*(n+1)/2)*(n*(n+1)/2))
      (by change (n*N/2)*(n*N/2) ≤ 8*N^4; omega)
    have wTwice := width (2*n) (by dsimp [N] at *; omega)
    have wTwiceNext := width (2*n+1) (by omega)
    have wTriple := width (n*(n+1)*(2*n+1)) (by change n*N*(2*n+1) ≤ 8*N^4; omega)
    have wDivided := width (n*(n+1)*(2*n+1)/6) (by change n*N*(2*n+1)/6 ≤ 8*N^4; omega)
    simp only [Int.natCast_add, Int.natCast_mul, Nat.cast_ofNat, Int.natCast_ediv] at wNext wProduct wHalf wSquare wTwice wTwiceNext wTriple wDivided
    have wOne : PrimitiveWork.bitLength (1 : Int) ≤ 8*L := by change 1 ≤ 8*L; omega
    have wTwo : PrimitiveWork.bitLength (2 : Int) ≤ 8*L := by change 2 ≤ 8*L; omega
    have wSix : PrimitiveWork.bitLength (6 : Int) ≤ 8*L := by change 3 ≤ 8*L; omega
    have paired (a b : Nat) (ha : a ≤ 8*L) (hb : b ≤ 8*L) :
        a*b ≤ 64*L^2 := by
      have multiplied := Nat.mul_le_mul ha hb
      nlinarith only [multiplied]
    have pProduct := paired _ _ wN wNext
    have pHalf := paired _ _ wProduct wTwo
    have pSquare := paired _ _ wHalf wHalf
    have pTwice := paired _ _ wTwo wN
    have pTriple := paired _ _ wProduct wTwiceNext
    have pDivided := paired _ _ wTriple wSix
    have logSquare : L ≤ L^2 := by nlinarith only [logPositive]
    simp only [List.length_cons, List.length_nil]
    change _ ≤ 4096 * L^2
    simp only [show PrimitiveWork.bitLength (1 : Int) = 1 from rfl,
      show PrimitiveWork.bitLength (2 : Int) = 2 from rfl,
      show PrimitiveWork.bitLength (6 : Int) = 3 from rfl] at *
    nlinarith only [pProduct, pHalf, pSquare, pTwice, pTriple, pDivided,
      wN, wTwice, wSquare, wDivided, logPositive, logSquare]

theorem fixed_bound (n : Nat) :
    Calls.Runtime.TimeBound .word Solvers.P6Ast.module "solve" [.int n] 128 := by
  apply Exists.intro
  refine ⟨.ok (.int (((n : Int) * (n + 1) / 2) * (n * (n + 1) / 2) -
    n * (n + 1) * (2 * n + 1) / 6)), ?_, ?_⟩
  · repeat first
      | exact RuntimeCost.Finishes.done rfl rfl
      | apply RuntimeCost.Finishes.next <;> first | rfl | skip
  · dsimp
    decide

theorem bigO : Calls.Runtime.IsBigO .bits Solvers.P6Ast.module "solve"
    (fun n : Nat => [.int n]) (fun _ => True) id (fun n => (Nat.log2 (n+1)+1)^2) := by
  exact ⟨4096, 0, fun n _ _ => source_bound n⟩

end solvers.«6».runtimeCost

theorem solvers.«6».solve.«BigO(1)» :
    FormalSW.PyAst.Calls.Runtime.IsBigO .word Solvers.P6Ast.module "solve"
      (fun n : Nat => [.int n])
      (fun n => FormalSW.PyAst.Calls.Condition Solvers.P6Ast.module
        (fun name => if name = "n" then some (.int n) else none) Solvers.P6Ast.requires_solve_0)
      id (fun _ => 1) := by
  exact ⟨128, 0, fun n _ _ => solvers.«6».runtimeCost.fixed_bound n⟩
