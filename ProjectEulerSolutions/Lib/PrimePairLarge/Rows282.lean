import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9127_small : List Nat := []
theorem row9127_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9127 row9127_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9127_block_005 : List Nat := [9187, 9343, 9547, 9679, 10273]
def plan9127_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .second, (.firstFactor 11), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 13), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9127_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9127 row9127_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9127_block_005
  · decide
  · rfl

def row9127_block_006 : List Nat := [10597, 10891, 11677, 12379]
def plan9127_block_006 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 11), (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), .second, .first, .first, .first, .second, .first, .first, .first, .possible, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .second, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 29), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9127_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9127 row9127_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9127_block_006
  · decide
  · rfl

def row9127_block_007 : List Nat := [12613, 13933, 14389]
def plan9127_block_007 : List Budget.Choice := [.first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 13), .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 37), .first, (.secondFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), (.firstFactor 11), .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .possible, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row9127_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9127 row9127_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9127_block_007
  · decide
  · rfl

def row9127_block_008 : List Nat := [15511, 15823]
def plan9127_block_008 : List Budget.Choice := [(.firstFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 37), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .possible, (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 37), (.firstFactor 19), (.firstFactor 17), (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), (.secondFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9127_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9127 row9127_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9127_block_008
  · decide
  · rfl

def row9127_block_009 : List Nat := [16603]
def plan9127_block_009 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 23), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 19), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9127_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9127 row9127_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9127_block_009
  · decide
  · rfl

def row9127_block_010 : List Nat := []
theorem row9127_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9127 row9127_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9127_block_011 : List Nat := []
theorem row9127_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9127 row9127_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9127_block_012 : List Nat := []
theorem row9127_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9127 row9127_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9127_block_013 : List Nat := []
theorem row9127_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9127 row9127_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9127_large_left_left : List Nat := row9127_small ++ row9127_block_005
theorem row9127_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9127 row9127_large_left_left :=
  Budget.RowUpperBound.branch row9127_small_covers row9127_block_005_covers

def row9127_large_left_right_right : List Nat := row9127_block_007 ++ row9127_block_008
theorem row9127_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9127 row9127_large_left_right_right :=
  Budget.RowUpperBound.branch row9127_block_007_covers row9127_block_008_covers

def row9127_large_left_right : List Nat := row9127_block_006 ++ row9127_large_left_right_right
theorem row9127_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9127 row9127_large_left_right :=
  Budget.RowUpperBound.branch row9127_block_006_covers row9127_large_left_right_right_covers

def row9127_large_left : List Nat := row9127_large_left_left ++ row9127_large_left_right
theorem row9127_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9127 row9127_large_left :=
  Budget.RowUpperBound.branch row9127_large_left_left_covers row9127_large_left_right_covers

def row9127_large_right_left : List Nat := row9127_block_009 ++ row9127_block_010
theorem row9127_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9127 row9127_large_right_left :=
  Budget.RowUpperBound.branch row9127_block_009_covers row9127_block_010_covers

def row9127_large_right_right_right : List Nat := row9127_block_012 ++ row9127_block_013
theorem row9127_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9127 row9127_large_right_right_right :=
  Budget.RowUpperBound.branch row9127_block_012_covers row9127_block_013_covers

def row9127_large_right_right : List Nat := row9127_block_011 ++ row9127_large_right_right_right
theorem row9127_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9127 row9127_large_right_right :=
  Budget.RowUpperBound.branch row9127_block_011_covers row9127_large_right_right_right_covers

def row9127_large_right : List Nat := row9127_large_right_left ++ row9127_large_right_right
theorem row9127_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9127 row9127_large_right :=
  Budget.RowUpperBound.branch row9127_large_right_left_covers row9127_large_right_right_covers

def row9127_large : List Nat := row9127_large_left ++ row9127_large_right
theorem row9127_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9127 row9127_large :=
  Budget.RowUpperBound.branch row9127_large_left_covers row9127_large_right_covers

def row9127 : List Nat := row9127_large
theorem row9127_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9127 row9127 := row9127_large_covers

def row9133_small : List Nat := []
theorem row9133_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9133 row9133_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9133_block_005 : List Nat := [9319, 9337, 9397, 9901, 10039]
def plan9133_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .possible, .first, .possible, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 11), .possible, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .second, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 37), .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.secondFactor 31), .first, .second, .first, (.firstFactor 19), .first, .second, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row9133_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9133 row9133_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9133_block_005
  · decide
  · rfl

def row9133_block_006 : List Nat := [10429, 11353, 11527]
def plan9133_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .second, .first, .second, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 23), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 13), .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 17), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row9133_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9133 row9133_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9133_block_006
  · decide
  · rfl

def row9133_block_007 : List Nat := [12619, 13441, 14323]
def plan9133_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 29), .first, .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 17), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 31), .second, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 13), .second, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row9133_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9133 row9133_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9133_block_007
  · decide
  · rfl

def row9133_block_008 : List Nat := [15889, 16333]
def plan9133_block_008 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 29), .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 29), (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 11), (.secondFactor 7), (.secondFactor 19), .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, .possible, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .possible, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9133_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9133 row9133_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9133_block_008
  · decide
  · rfl

def row9133_block_009 : List Nat := []
def plan9133_block_009 : List Budget.Choice := [(.secondFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 31), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .second, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9133_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9133 row9133_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9133_block_009
  · decide
  · rfl

def row9133_block_010 : List Nat := []
theorem row9133_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9133 row9133_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9133_block_011 : List Nat := []
theorem row9133_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9133 row9133_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9133_block_012 : List Nat := []
theorem row9133_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9133 row9133_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9133_block_013 : List Nat := []
theorem row9133_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9133 row9133_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9133_large_left_left : List Nat := row9133_small ++ row9133_block_005
theorem row9133_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9133 row9133_large_left_left :=
  Budget.RowUpperBound.branch row9133_small_covers row9133_block_005_covers

def row9133_large_left_right_right : List Nat := row9133_block_007 ++ row9133_block_008
theorem row9133_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9133 row9133_large_left_right_right :=
  Budget.RowUpperBound.branch row9133_block_007_covers row9133_block_008_covers

def row9133_large_left_right : List Nat := row9133_block_006 ++ row9133_large_left_right_right
theorem row9133_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9133 row9133_large_left_right :=
  Budget.RowUpperBound.branch row9133_block_006_covers row9133_large_left_right_right_covers

def row9133_large_left : List Nat := row9133_large_left_left ++ row9133_large_left_right
theorem row9133_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9133 row9133_large_left :=
  Budget.RowUpperBound.branch row9133_large_left_left_covers row9133_large_left_right_covers

def row9133_large_right_left : List Nat := row9133_block_009 ++ row9133_block_010
theorem row9133_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9133 row9133_large_right_left :=
  Budget.RowUpperBound.branch row9133_block_009_covers row9133_block_010_covers

def row9133_large_right_right_right : List Nat := row9133_block_012 ++ row9133_block_013
theorem row9133_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9133 row9133_large_right_right_right :=
  Budget.RowUpperBound.branch row9133_block_012_covers row9133_block_013_covers

def row9133_large_right_right : List Nat := row9133_block_011 ++ row9133_large_right_right_right
theorem row9133_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9133 row9133_large_right_right :=
  Budget.RowUpperBound.branch row9133_block_011_covers row9133_large_right_right_right_covers

def row9133_large_right : List Nat := row9133_large_right_left ++ row9133_large_right_right
theorem row9133_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9133 row9133_large_right :=
  Budget.RowUpperBound.branch row9133_large_right_left_covers row9133_large_right_right_covers

def row9133_large : List Nat := row9133_large_left ++ row9133_large_right
theorem row9133_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9133 row9133_large :=
  Budget.RowUpperBound.branch row9133_large_left_covers row9133_large_right_covers

def row9133 : List Nat := row9133_large
theorem row9133_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9133 row9133 := row9133_large_covers

def row9137_small : List Nat := []
theorem row9137_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9137 row9137_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9137_block_005 : List Nat := [9293, 9689, 10331]
def plan9137_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.secondFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 29), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9137_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9137 row9137_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9137_block_005
  · decide
  · rfl

def row9137_block_006 : List Nat := [10799, 11243, 12377]
def plan9137_block_006 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 11), .first, .possible, .first, .first, (.secondFactor 19), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.secondFactor 11), .first, .second, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9137_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9137 row9137_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9137_block_006
  · decide
  · rfl

def row9137_block_007 : List Nat := [12479, 12791, 12959, 13001, 14033]
def plan9137_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 31), (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 19), .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 31), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .possible, (.secondFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .second, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9137_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9137 row9137_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9137_block_007
  · decide
  · rfl

def row9137_block_008 : List Nat := [15017, 15329]
def plan9137_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 37), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 31), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .second, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9137_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9137 row9137_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9137_block_008
  · decide
  · rfl

def row9137_block_009 : List Nat := [16547]
def plan9137_block_009 : List Budget.Choice := [.first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9137_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9137 row9137_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9137_block_009
  · decide
  · rfl

def row9137_block_010 : List Nat := []
theorem row9137_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9137 row9137_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9137_block_011 : List Nat := []
theorem row9137_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9137 row9137_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9137_block_012 : List Nat := []
theorem row9137_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9137 row9137_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9137_block_013 : List Nat := []
theorem row9137_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9137 row9137_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9137_large_left_left : List Nat := row9137_small ++ row9137_block_005
theorem row9137_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9137 row9137_large_left_left :=
  Budget.RowUpperBound.branch row9137_small_covers row9137_block_005_covers

def row9137_large_left_right_right : List Nat := row9137_block_007 ++ row9137_block_008
theorem row9137_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9137 row9137_large_left_right_right :=
  Budget.RowUpperBound.branch row9137_block_007_covers row9137_block_008_covers

def row9137_large_left_right : List Nat := row9137_block_006 ++ row9137_large_left_right_right
theorem row9137_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9137 row9137_large_left_right :=
  Budget.RowUpperBound.branch row9137_block_006_covers row9137_large_left_right_right_covers

def row9137_large_left : List Nat := row9137_large_left_left ++ row9137_large_left_right
theorem row9137_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9137 row9137_large_left :=
  Budget.RowUpperBound.branch row9137_large_left_left_covers row9137_large_left_right_covers

def row9137_large_right_left : List Nat := row9137_block_009 ++ row9137_block_010
theorem row9137_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9137 row9137_large_right_left :=
  Budget.RowUpperBound.branch row9137_block_009_covers row9137_block_010_covers

def row9137_large_right_right_right : List Nat := row9137_block_012 ++ row9137_block_013
theorem row9137_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9137 row9137_large_right_right_right :=
  Budget.RowUpperBound.branch row9137_block_012_covers row9137_block_013_covers

def row9137_large_right_right : List Nat := row9137_block_011 ++ row9137_large_right_right_right
theorem row9137_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9137 row9137_large_right_right :=
  Budget.RowUpperBound.branch row9137_block_011_covers row9137_large_right_right_right_covers

def row9137_large_right : List Nat := row9137_large_right_left ++ row9137_large_right_right
theorem row9137_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9137 row9137_large_right :=
  Budget.RowUpperBound.branch row9137_large_right_left_covers row9137_large_right_right_covers

def row9137_large : List Nat := row9137_large_left ++ row9137_large_right
theorem row9137_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9137 row9137_large :=
  Budget.RowUpperBound.branch row9137_large_left_covers row9137_large_right_covers

def row9137 : List Nat := row9137_large
theorem row9137_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9137 row9137 := row9137_large_covers

def row9151_small : List Nat := []
theorem row9151_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9151 row9151_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9151_block_005 : List Nat := [9643, 9739, 9949]
def plan9151_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 29), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), .second, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .possible, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .second, .first, (.firstFactor 17), (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .second, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.secondFactor 17), .first, .possible, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .second]
set_option maxRecDepth 2048 in
theorem row9151_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9151 row9151_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9151_block_005
  · decide
  · rfl

def row9151_block_006 : List Nat := [10711, 10789, 11311, 11731, 11863]
def plan9151_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .second, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, .second, (.secondFactor 19), .first, .possible, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .possible, .second, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 19), .possible, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row9151_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9151 row9151_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9151_block_006
  · decide
  · rfl

def row9151_block_007 : List Nat := [12763, 12829, 13063, 13687, 13921, 14449]
def plan9151_block_007 : List Budget.Choice := [(.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .possible, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 37), .possible, (.secondFactor 7), (.secondFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .possible, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .second, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 23), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 37), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row9151_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9151 row9151_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9151_block_007
  · decide
  · rfl

def row9151_block_008 : List Nat := [15199, 15679, 16087, 16183]
def plan9151_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, (.secondFactor 19), .first, .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .second, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), .first, .possible, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9151_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9151 row9151_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9151_block_008
  · decide
  · rfl

def row9151_block_009 : List Nat := [16651, 16759]
def plan9151_block_009 : List Budget.Choice := [(.secondFactor 13), .first, .first, .first, (.secondFactor 17), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9151_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9151 row9151_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9151_block_009
  · decide
  · rfl

def row9151_block_010 : List Nat := []
theorem row9151_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9151 row9151_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9151_block_011 : List Nat := []
theorem row9151_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9151 row9151_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9151_block_012 : List Nat := []
theorem row9151_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9151 row9151_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9151_block_013 : List Nat := []
theorem row9151_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9151 row9151_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9151_large_left_left : List Nat := row9151_small ++ row9151_block_005
theorem row9151_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9151 row9151_large_left_left :=
  Budget.RowUpperBound.branch row9151_small_covers row9151_block_005_covers

def row9151_large_left_right_right : List Nat := row9151_block_007 ++ row9151_block_008
theorem row9151_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9151 row9151_large_left_right_right :=
  Budget.RowUpperBound.branch row9151_block_007_covers row9151_block_008_covers

def row9151_large_left_right : List Nat := row9151_block_006 ++ row9151_large_left_right_right
theorem row9151_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9151 row9151_large_left_right :=
  Budget.RowUpperBound.branch row9151_block_006_covers row9151_large_left_right_right_covers

def row9151_large_left : List Nat := row9151_large_left_left ++ row9151_large_left_right
theorem row9151_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9151 row9151_large_left :=
  Budget.RowUpperBound.branch row9151_large_left_left_covers row9151_large_left_right_covers

def row9151_large_right_left : List Nat := row9151_block_009 ++ row9151_block_010
theorem row9151_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9151 row9151_large_right_left :=
  Budget.RowUpperBound.branch row9151_block_009_covers row9151_block_010_covers

def row9151_large_right_right_right : List Nat := row9151_block_012 ++ row9151_block_013
theorem row9151_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9151 row9151_large_right_right_right :=
  Budget.RowUpperBound.branch row9151_block_012_covers row9151_block_013_covers

def row9151_large_right_right : List Nat := row9151_block_011 ++ row9151_large_right_right_right
theorem row9151_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9151 row9151_large_right_right :=
  Budget.RowUpperBound.branch row9151_block_011_covers row9151_large_right_right_right_covers

def row9151_large_right : List Nat := row9151_large_right_left ++ row9151_large_right_right
theorem row9151_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9151 row9151_large_right :=
  Budget.RowUpperBound.branch row9151_large_right_left_covers row9151_large_right_right_covers

def row9151_large : List Nat := row9151_large_left ++ row9151_large_right
theorem row9151_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9151 row9151_large :=
  Budget.RowUpperBound.branch row9151_large_left_covers row9151_large_right_covers

def row9151 : List Nat := row9151_large
theorem row9151_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9151 row9151 := row9151_large_covers

end ProjectEulerSolutions.PrimePairLarge
