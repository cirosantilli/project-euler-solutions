import Mathlib.Data.List.ProdSigma
import Mathlib.Data.List.Range
import Mathlib.Data.Int.Bitwise

/-! Weighted XOR decryption, independent of the old P59 search and termination files. -/
namespace ProjectEulerSolutions.XorDecryption
set_option Elab.async false

abbrev Key := Int × Int × Int

def encodeBytes (cipher : List UInt8) : List Int := cipher.map fun byte => (byte.toNat : Int)

def lowercaseCode (value : Int) : Int := if 65 ≤ value ∧ value ≤ 90 then value + 32 else value

/-- The exact source weights, including apostrophe 39 and the eight penalties. -/
def charWeight (value : Int) : Int :=
  let code := lowercaseCode value
  if code = 32 then 10
  else if code = 101 then 6
  else if code ∈ [116, 97, 111] then 5
  else if code ∈ [105, 110, 115, 104, 114] then 4
  else if code ∈ [100, 108] then 3
  else if code ∈ [117, 99, 109, 119, 102, 103, 121, 112] then 2
  else if code ∈ [98, 118, 107] then 1
  else if code ∈ [46, 44, 59, 58, 39, 34, 33, 63, 40, 41, 45] then 1
  else if code ∈ [123, 125, 91, 93, 124, 94, 126, 96] then -3
  else 0

def printable (value : Int) : Bool := decide (value = 10 ∨ value = 13 ∨ (32 ≤ value ∧ value ≤ 126))

def keyValues (key : Key) : List Int := [key.1, key.2.1, key.2.2]

def keyByte (key : Key) (position : Nat) : Int :=
  if position % 3 = 0 then key.1 else if position % 3 = 1 then key.2.1 else key.2.2

def decryptFrom (key : Key) : Nat → List Int → List Int
  | _, [] => []
  | position, head :: tail => Int.xor head (keyByte key position) :: decryptFrom key (position + 1) tail

def decrypt (cipher : List Int) (key : Key) : List Int := decryptFrom key 0 cipher

/-- The materialized reference validates the whole text and returns its score and sum. -/
def scoreText (text : List Int) : Option (Int × Int) :=
  if text.all printable then some ((text.map charWeight).sum, text.sum) else none

def scoreOnly (text : List Int) : Option Int := (scoreText text).map Prod.fst

/-- The root decodes and scores in source order, rejecting immediately on a bad byte. -/
def scoreStreaming (key : Key) : Nat → List Int → Int → Int → Option (Int × Int)
  | _, [], score, total => some (score, total)
  | position, head :: tail, score, total =>
      let plain := Int.xor head (keyByte key position)
      if printable plain then scoreStreaming key (position + 1) tail (score + charWeight plain) (total + plain)
      else none

def scoreKey (cipher : List Int) (key : Key) : Option (Int × Int) := scoreText (decrypt cipher key)
def scoreKeyStreaming (cipher : List Int) (key : Key) : Option (Int × Int) := scoreStreaming key 0 cipher 0 0

def letters : List Int := (List.range' 97 26).map fun value => (value : Int)

/-- Product order is exactly a-outermost, then b, then c, without expanding 26^3 keys. -/
def keysFor (alphabet : List Int) : List Key :=
  ((alphabet.product alphabet).product alphabet).map fun entry => (entry.1.1, entry.1.2, entry.2)

def keys : List Key := keysFor letters

structure Selection where
  key : Key
  score : Int
  total : Int
  deriving DecidableEq

/-- None candidates are ignored; equal scores retain the earlier incumbent. -/
def choose : Option Selection → Option Selection → Option Selection
  | best, none => best
  | none, some candidate => some candidate
  | some best, some candidate => if best.score < candidate.score then some candidate else some best

def advance (best : Option Selection) (key : Key) (result : Option (Int × Int)) : Option Selection :=
  choose best (result.map fun pair => ⟨key, pair.1, pair.2⟩)

def scan (evaluate : Key → Option (Int × Int)) (space : List Key) (initial : Option Selection) : Option Selection :=
  space.foldl (fun best key => advance best key (evaluate key)) initial

def best (cipher : List Int) : Option Selection := scan (scoreKey cipher) keys none

def selectedSum (selection : Option Selection) : Int := (selection.map Selection.total).getD 0
def selectedKey (selection : Option Selection) : List Int := (selection.map fun item => keyValues item.key).getD []

def answer (cipher : List UInt8) : Int := selectedSum (best (encodeBytes cipher))


@[simp] theorem decryptFrom_length (key : Key) (position : Nat) (cipher : List Int) :
    (decryptFrom key position cipher).length = cipher.length := by
  induction cipher generalizing position with
  | nil => rfl
  | cons head tail ih => simp only [decryptFrom, List.length_cons, ih]

/-- A successful score always contains the sum of the entire plaintext. -/
theorem scoreText_total {text : List Int} {score total : Int}
    (computed : scoreText text = some (score, total)) : total = text.sum := by
  unfold scoreText at computed
  split at computed
  · have same : (text.map charWeight).sum = score ∧ text.sum = total := by simpa using computed
    exact same.2.symm
  · contradiction

/-- Fusion preserves accumulated totals and the exact invalid-text outcome. -/
theorem scoreStreaming_eq (key : Key) (position : Nat) (cipher : List Int) (score total : Int) :
    scoreStreaming key position cipher score total =
      (scoreText (decryptFrom key position cipher)).map fun result => (score + result.1, total + result.2) := by
  induction cipher generalizing position score total with
  | nil => simp [scoreStreaming, decryptFrom, scoreText]
  | cons head tail ih =>
      cases first : printable (Int.xor head (keyByte key position)) <;>
        cases rest : (decryptFrom key (position + 1) tail).all printable <;>
          simp [scoreStreaming, decryptFrom, scoreText, first, rest, ih, Int.add_assoc]

theorem scoreKeyStreaming_eq (cipher : List Int) (key : Key) : scoreKeyStreaming cipher key = scoreKey cipher key := by
  simpa [scoreKeyStreaming, scoreKey, decrypt] using scoreStreaming_eq key 0 cipher 0 0

@[simp] theorem letters_length : letters.length = 26 := by simp [letters]

theorem keysFor_length (alphabet : List Int) : (keysFor alphabet).length = alphabet.length ^ 3 := by
  simp only [keysFor, List.length_map]
  calc
    _ = (alphabet.product alphabet).length * alphabet.length := List.length_product _ _
    _ = (alphabet.length * alphabet.length) * alphabet.length :=
      congrArg (fun size => size * alphabet.length) (List.length_product alphabet alphabet)
    _ = alphabet.length ^ 3 := by simp only [pow_succ, pow_zero, one_mul]

theorem keys_length : keys.length = 17576 := by
  rw [keys, keysFor_length, letters_length]
  decide

@[simp] theorem choose_none (best : Option Selection) : choose best none = best := by cases best <;> rfl

@[simp] theorem choose_first (candidate : Selection) : choose none (some candidate) = some candidate := rfl

theorem choose_keeps_tie (best candidate : Selection) (same : candidate.score = best.score) :
    choose (some best) (some candidate) = some best := by simp [choose, same]

theorem scan_append (evaluate : Key → Option (Int × Int)) (front back : List Key) (initial : Option Selection) :
    scan evaluate (front ++ back) initial = scan evaluate back (scan evaluate front initial) := by
  exact List.foldl_append

/-- Different source scorers may be substituted when they agree on every enumerated key. -/
theorem scan_congr (left right : Key → Option (Int × Int)) (space : List Key)
    (same : ∀ key ∈ space, left key = right key) (initial : Option Selection) :
    scan left space initial = scan right space initial := by
  induction space generalizing initial with
  | nil => rfl
  | cons key tail ih =>
      simp only [scan, List.foldl_cons]
      rw [same key (List.mem_cons_self)]
      exact ih (fun other present => same other (List.mem_cons_of_mem key present)) _

theorem scan_streaming_eq (cipher : List Int) (space : List Key) (initial : Option Selection) :
    scan (scoreKeyStreaming cipher) space initial = scan (scoreKey cipher) space initial :=
  scan_congr _ _ space (fun key _ => scoreKeyStreaming_eq cipher key) initial

/-- Generic product folding proves the nested-loop connection without key-space expansion. -/
theorem foldl_product {α β σ : Type*} (xs : List α) (ys : List β) (update : σ → α × β → σ) (initial : σ) :
    (xs.product ys).foldl update initial =
      xs.foldl (fun state left => ys.foldl (fun current right => update current (left, right)) state) initial := by
  induction xs generalizing initial with
  | nil => rfl
  | cons head tail ih =>
      change ((ys.map fun right => (head, right)) ++ tail.product ys).foldl update initial = _
      rw [List.foldl_append, List.foldl_map]
      exact ih _

theorem scan_keysFor_nested (evaluate : Key → Option (Int × Int)) (alphabet : List Int) (initial : Option Selection) :
    scan evaluate (keysFor alphabet) initial =
      alphabet.foldl (fun outer a => alphabet.foldl (fun middle b => alphabet.foldl
        (fun inner c => advance inner (a, b, c) (evaluate (a, b, c))) middle) outer) initial := by
  simp only [scan, keysFor, List.foldl_map, foldl_product]

/-- Rejected candidates never manufacture a partial-prefix sum. -/
theorem scan_none (evaluate : Key → Option (Int × Int)) (space : List Key)
    (rejected : ∀ key ∈ space, evaluate key = none) : scan evaluate space none = none := by
  induction space with
  | nil => rfl
  | cons key tail ih =>
      simp only [scan, List.foldl_cons]
      rw [rejected key List.mem_cons_self]
      exact ih (fun other present => rejected other (List.mem_cons_of_mem key present))

@[simp] theorem selectedSum_none : selectedSum none = 0 := rfl
@[simp] theorem selectedKey_none : selectedKey none = [] := rfl


@[simp] theorem keyValues_length (key : Key) : (keyValues key).length = 3 := rfl

/-- The reference's literal list subscript agrees with the root's three branches. -/
theorem keyValues_getElem?_mod (key : Key) (position : Nat) :
    (keyValues key)[position % 3]? = some (keyByte key position) := by
  have bound : position % 3 < 3 := Nat.mod_lt _ (by decide)
  by_cases zero : position % 3 = 0
  · simp only [keyValues, keyByte, zero, ↓reduceIte, List.getElem?_cons_zero]
  · by_cases one : position % 3 = 1
    · simp [keyValues, keyByte, one]
    · have two : position % 3 = 2 := by omega
      simp [keyValues, keyByte, two]

@[simp] theorem scoreOnly_nil : scoreOnly [] = some 0 := rfl

/-- The source text_score loop can return immediately on the first rejected byte. -/
theorem scoreOnly_cons (value : Int) (rest : List Int) :
    scoreOnly (value :: rest) =
      if printable value then (scoreOnly rest).map (fun score => charWeight value + score)
      else none := by
  cases first : printable value <;> cases following : rest.all printable <;>
    simp [scoreOnly, scoreText, first, following]

/-- Reconstruct the reference's later sum(text) from the optional score result. -/
theorem scoreText_eq_scoreOnly (text : List Int) :
    scoreText text = (scoreOnly text).map (fun score => (score, text.sum)) := by
  cases valid : text.all printable <;> simp [scoreOnly, scoreText, valid]

theorem scoreOnly_eq_some_iff {text : List Int} {score : Int} :
    scoreOnly text = some score ↔ text.all printable = true ∧ score = (text.map charWeight).sum := by
  cases valid : text.all printable <;> simp [scoreOnly, scoreText, valid, eq_comm]

theorem scoreOnly_eq_none_iff (text : List Int) :
    scoreOnly text = none ↔ text.all printable = false := by
  cases valid : text.all printable <;> simp [scoreOnly, scoreText, valid]

/-- Generic materialization lemma; it does not enumerate the key space. -/
theorem map_product_eq_flatMap {α β γ : Type*} (xs : List α) (ys : List β) (f : α × β → γ) :
    (xs.product ys).map f = xs.flatMap (fun a => ys.map (fun b => f (a, b))) := by
  induction xs with
  | nil => rfl
  | cons head tail ih =>
      change ((ys.map (fun right => (head, right))) ++ tail.product ys).map f = _
      simp only [List.map_append, List.map_map, List.flatMap_cons, ih, Function.comp_def]

theorem flatMap_product_eq_flatMap {α β γ : Type*} (xs : List α) (ys : List β) (f : α × β → List γ) :
    (xs.product ys).flatMap f = xs.flatMap (fun a => ys.flatMap (fun b => f (a, b))) := by
  induction xs with
  | nil => rfl
  | cons head tail ih =>
      change ((ys.map (fun right => (head, right))) ++ tail.product ys).flatMap f = _
      simp only [List.flatMap_append, List.flatMap_map, List.flatMap_cons, ih]

/-- Actual reference key buffers occur in exactly this lexicographic order. -/
theorem keysFor_map_keyValues (alphabet : List Int) :
    (keysFor alphabet).map keyValues =
      alphabet.flatMap (fun a => alphabet.flatMap (fun b => alphabet.map (fun c => [a, b, c]))) := by
  simp only [keysFor, List.map_map]
  rw [map_product_eq_flatMap, flatMap_product_eq_flatMap]
  rfl

theorem encodeBytes_bounds {cipher : List UInt8} {value : Int} (present : value ∈ encodeBytes cipher) :
    0 ≤ value ∧ value < 256 := by
  obtain ⟨byte, _, rfl⟩ := List.mem_map.mp present
  have bound := byte.toNat_lt
  constructor <;> omega

theorem letters_bounds {value : Int} (present : value ∈ letters) : 97 ≤ value ∧ value ≤ 122 := by
  change value ∈ (List.range' 97 26).map (fun natural : Nat => (natural : Int)) at present
  obtain ⟨natural, inRange, rfl⟩ :=
    (List.mem_map (f := fun natural : Nat => (natural : Int))).mp present
  have bounds : 97 ≤ natural ∧ natural < 97 + 26 := List.mem_range'_1.mp inRange
  constructor <;> omega

/-- No signed or out-of-bounds array index is produced when XORing byte values. -/
theorem xor_byte_bounds {left right : Int} (leftBounds : 0 ≤ left ∧ left < 256)
    (rightBounds : 0 ≤ right ∧ right < 256) :
    0 ≤ Int.xor left right ∧ Int.xor left right < 256 := by
  have leftCast : (left.toNat : Int) = left := Int.toNat_of_nonneg leftBounds.1
  have rightCast : (right.toNat : Int) = right := Int.toNat_of_nonneg rightBounds.1
  have leftNat : left.toNat < 2 ^ 8 := by omega
  have rightNat : right.toNat < 2 ^ 8 := by omega
  have xorNat : left.toNat ^^^ right.toNat < 2 ^ 8 :=
    Nat.bitwise_lt_two_pow leftNat rightNat
  rw [← leftCast, ← rightCast]
  change 0 ≤ ((left.toNat ^^^ right.toNat : Nat) : Int) ∧
    ((left.toNat ^^^ right.toNat : Nat) : Int) < 256
  constructor <;> omega

end ProjectEulerSolutions.XorDecryption
