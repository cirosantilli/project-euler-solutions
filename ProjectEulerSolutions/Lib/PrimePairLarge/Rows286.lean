import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9257_small : List Nat := []
theorem row9257_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9257 row9257_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9257_block_005 : List Nat := [9371, 9749]
def plan9257_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 29), .first, .second, (.firstFactor 11), .first, .second, .first, (.secondFactor 19), .first, .first, .possible, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 23), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .possible, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .second, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .second, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row9257_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9257 row9257_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9257_block_005
  · decide
  · rfl

def row9257_block_006 : List Nat := [11261, 11519, 11903, 11909]
def plan9257_block_006 : List Budget.Choice := [(.secondFactor 13), .first, (.secondFactor 31), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .second, (.secondFactor 29), (.secondFactor 13), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 31), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .possible, (.secondFactor 17), (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .possible, .possible, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .second, (.firstFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9257_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9257 row9257_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9257_block_006
  · decide
  · rfl

def row9257_block_007 : List Nat := [13163, 13883, 14081]
def plan9257_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 13), (.secondFactor 31), (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 23), (.secondFactor 7), .first, .possible, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9257_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9257 row9257_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9257_block_007
  · decide
  · rfl

def row9257_block_008 : List Nat := [15083, 16229]
def plan9257_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 13), .second, .first, .second, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 29), (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 37), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9257_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9257 row9257_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9257_block_008
  · decide
  · rfl

def row9257_block_009 : List Nat := []
def plan9257_block_009 : List Budget.Choice := [.first, .second, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9257_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9257 row9257_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9257_block_009
  · decide
  · rfl

def row9257_block_010 : List Nat := []
theorem row9257_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9257 row9257_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9257_block_011 : List Nat := []
theorem row9257_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9257 row9257_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9257_block_012 : List Nat := []
theorem row9257_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9257 row9257_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9257_block_013 : List Nat := []
theorem row9257_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9257 row9257_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9257_large_left_left : List Nat := row9257_small ++ row9257_block_005
theorem row9257_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9257 row9257_large_left_left :=
  Budget.RowUpperBound.branch row9257_small_covers row9257_block_005_covers

def row9257_large_left_right_right : List Nat := row9257_block_007 ++ row9257_block_008
theorem row9257_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9257 row9257_large_left_right_right :=
  Budget.RowUpperBound.branch row9257_block_007_covers row9257_block_008_covers

def row9257_large_left_right : List Nat := row9257_block_006 ++ row9257_large_left_right_right
theorem row9257_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9257 row9257_large_left_right :=
  Budget.RowUpperBound.branch row9257_block_006_covers row9257_large_left_right_right_covers

def row9257_large_left : List Nat := row9257_large_left_left ++ row9257_large_left_right
theorem row9257_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9257 row9257_large_left :=
  Budget.RowUpperBound.branch row9257_large_left_left_covers row9257_large_left_right_covers

def row9257_large_right_left : List Nat := row9257_block_009 ++ row9257_block_010
theorem row9257_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9257 row9257_large_right_left :=
  Budget.RowUpperBound.branch row9257_block_009_covers row9257_block_010_covers

def row9257_large_right_right_right : List Nat := row9257_block_012 ++ row9257_block_013
theorem row9257_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9257 row9257_large_right_right_right :=
  Budget.RowUpperBound.branch row9257_block_012_covers row9257_block_013_covers

def row9257_large_right_right : List Nat := row9257_block_011 ++ row9257_large_right_right_right
theorem row9257_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9257 row9257_large_right_right :=
  Budget.RowUpperBound.branch row9257_block_011_covers row9257_large_right_right_right_covers

def row9257_large_right : List Nat := row9257_large_right_left ++ row9257_large_right_right
theorem row9257_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9257 row9257_large_right :=
  Budget.RowUpperBound.branch row9257_large_right_left_covers row9257_large_right_right_covers

def row9257_large : List Nat := row9257_large_left ++ row9257_large_right
theorem row9257_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9257 row9257_large :=
  Budget.RowUpperBound.branch row9257_large_left_covers row9257_large_right_covers

def row9257 : List Nat := row9257_large
theorem row9257_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9257 row9257 := row9257_large_covers

def row9277_small : List Nat := []
theorem row9277_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9277 row9277_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9277_block_005 : List Nat := [9283]
def plan9277_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 29), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.secondFactor 19), (.firstFactor 29), .first, .first, .second, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .second, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9277_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9277 row9277_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9277_block_005
  · decide
  · rfl

def row9277_block_006 : List Nat := [12301]
def plan9277_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .second, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row9277_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9277 row9277_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9277_block_006
  · decide
  · rfl

def row9277_block_007 : List Nat := [13411, 13729]
def plan9277_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 19), (.firstFactor 11), .second, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .possible, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 23), .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9277_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9277 row9277_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9277_block_007
  · decide
  · rfl

def row9277_block_008 : List Nat := [14869, 15013, 15643, 16417]
def plan9277_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .second, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 23), .first, (.firstFactor 37), .first, .second, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 19), (.secondFactor 11), (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9277_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9277 row9277_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9277_block_008
  · decide
  · rfl

def row9277_block_009 : List Nat := [16729]
def plan9277_block_009 : List Budget.Choice := [.first, .first, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .possible, (.firstFactor 7), (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9277_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9277 row9277_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9277_block_009
  · decide
  · rfl

def row9277_block_010 : List Nat := []
theorem row9277_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9277 row9277_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9277_block_011 : List Nat := []
theorem row9277_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9277 row9277_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9277_block_012 : List Nat := []
theorem row9277_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9277 row9277_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9277_block_013 : List Nat := []
theorem row9277_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9277 row9277_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9277_large_left_left : List Nat := row9277_small ++ row9277_block_005
theorem row9277_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9277 row9277_large_left_left :=
  Budget.RowUpperBound.branch row9277_small_covers row9277_block_005_covers

def row9277_large_left_right_right : List Nat := row9277_block_007 ++ row9277_block_008
theorem row9277_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9277 row9277_large_left_right_right :=
  Budget.RowUpperBound.branch row9277_block_007_covers row9277_block_008_covers

def row9277_large_left_right : List Nat := row9277_block_006 ++ row9277_large_left_right_right
theorem row9277_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9277 row9277_large_left_right :=
  Budget.RowUpperBound.branch row9277_block_006_covers row9277_large_left_right_right_covers

def row9277_large_left : List Nat := row9277_large_left_left ++ row9277_large_left_right
theorem row9277_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9277 row9277_large_left :=
  Budget.RowUpperBound.branch row9277_large_left_left_covers row9277_large_left_right_covers

def row9277_large_right_left : List Nat := row9277_block_009 ++ row9277_block_010
theorem row9277_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9277 row9277_large_right_left :=
  Budget.RowUpperBound.branch row9277_block_009_covers row9277_block_010_covers

def row9277_large_right_right_right : List Nat := row9277_block_012 ++ row9277_block_013
theorem row9277_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9277 row9277_large_right_right_right :=
  Budget.RowUpperBound.branch row9277_block_012_covers row9277_block_013_covers

def row9277_large_right_right : List Nat := row9277_block_011 ++ row9277_large_right_right_right
theorem row9277_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9277 row9277_large_right_right :=
  Budget.RowUpperBound.branch row9277_block_011_covers row9277_large_right_right_right_covers

def row9277_large_right : List Nat := row9277_large_right_left ++ row9277_large_right_right
theorem row9277_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9277 row9277_large_right :=
  Budget.RowUpperBound.branch row9277_large_right_left_covers row9277_large_right_right_covers

def row9277_large : List Nat := row9277_large_left ++ row9277_large_right
theorem row9277_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9277 row9277_large :=
  Budget.RowUpperBound.branch row9277_large_left_covers row9277_large_right_covers

def row9277 : List Nat := row9277_large
theorem row9277_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9277 row9277 := row9277_large_covers

def row9281_small : List Nat := []
theorem row9281_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9281 row9281_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9281_block_005 : List Nat := [9539, 10133]
def plan9281_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9281_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9281 row9281_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9281_block_005
  · decide
  · rfl

def row9281_block_006 : List Nat := []
def plan9281_block_006 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 23), .second, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 19), .first, .first, .second, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 37), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9281_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9281 row9281_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9281_block_006
  · decide
  · rfl

def row9281_block_007 : List Nat := [12641, 12821, 13829]
def plan9281_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 13), .second, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .possible, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 29), (.firstFactor 37), .first, .second, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 23), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 17), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row9281_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9281 row9281_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9281_block_007
  · decide
  · rfl

def row9281_block_008 : List Nat := [14519, 14951, 15137, 15263, 15581, 16073, 16451]
def plan9281_block_008 : List Budget.Choice := [.first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 7), .first, (.secondFactor 19), .second, .first, (.firstFactor 23), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 17), .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 31), .first, .first, .possible, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 19), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 11), .second, .second, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), .second, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row9281_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9281 row9281_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9281_block_008
  · decide
  · rfl

def row9281_block_009 : List Nat := []
def plan9281_block_009 : List Budget.Choice := [.first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9281_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9281 row9281_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9281_block_009
  · decide
  · rfl

def row9281_block_010 : List Nat := []
theorem row9281_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9281 row9281_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9281_block_011 : List Nat := []
theorem row9281_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9281 row9281_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9281_block_012 : List Nat := []
theorem row9281_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9281 row9281_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9281_block_013 : List Nat := []
theorem row9281_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9281 row9281_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9281_large_left_left : List Nat := row9281_small ++ row9281_block_005
theorem row9281_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9281 row9281_large_left_left :=
  Budget.RowUpperBound.branch row9281_small_covers row9281_block_005_covers

def row9281_large_left_right_right : List Nat := row9281_block_007 ++ row9281_block_008
theorem row9281_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9281 row9281_large_left_right_right :=
  Budget.RowUpperBound.branch row9281_block_007_covers row9281_block_008_covers

def row9281_large_left_right : List Nat := row9281_block_006 ++ row9281_large_left_right_right
theorem row9281_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9281 row9281_large_left_right :=
  Budget.RowUpperBound.branch row9281_block_006_covers row9281_large_left_right_right_covers

def row9281_large_left : List Nat := row9281_large_left_left ++ row9281_large_left_right
theorem row9281_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9281 row9281_large_left :=
  Budget.RowUpperBound.branch row9281_large_left_left_covers row9281_large_left_right_covers

def row9281_large_right_left : List Nat := row9281_block_009 ++ row9281_block_010
theorem row9281_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9281 row9281_large_right_left :=
  Budget.RowUpperBound.branch row9281_block_009_covers row9281_block_010_covers

def row9281_large_right_right_right : List Nat := row9281_block_012 ++ row9281_block_013
theorem row9281_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9281 row9281_large_right_right_right :=
  Budget.RowUpperBound.branch row9281_block_012_covers row9281_block_013_covers

def row9281_large_right_right : List Nat := row9281_block_011 ++ row9281_large_right_right_right
theorem row9281_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9281 row9281_large_right_right :=
  Budget.RowUpperBound.branch row9281_block_011_covers row9281_large_right_right_right_covers

def row9281_large_right : List Nat := row9281_large_right_left ++ row9281_large_right_right
theorem row9281_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9281 row9281_large_right :=
  Budget.RowUpperBound.branch row9281_large_right_left_covers row9281_large_right_right_covers

def row9281_large : List Nat := row9281_large_left ++ row9281_large_right
theorem row9281_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9281 row9281_large :=
  Budget.RowUpperBound.branch row9281_large_left_covers row9281_large_right_covers

def row9281 : List Nat := row9281_large
theorem row9281_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9281 row9281 := row9281_large_covers

def row9283_small : List Nat := []
theorem row9283_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9283 row9283_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9283_block_005 : List Nat := [9349, 9631]
def plan9283_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .possible, .first, .first, (.secondFactor 37), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 17), (.secondFactor 37), .first, .first, .first, .possible, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .second, (.secondFactor 23), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9283_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9283 row9283_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9283_block_005
  · decide
  · rfl

def row9283_block_006 : List Nat := [10399, 11113, 11161, 11497]
def plan9283_block_006 : List Budget.Choice := [.first, .possible, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .second, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 19), (.firstFactor 37), (.secondFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 7), .first, .possible, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .second, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 31), (.secondFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9283_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9283 row9283_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9283_block_006
  · decide
  · rfl

def row9283_block_007 : List Nat := [12841, 14281, 14449]
def plan9283_block_007 : List Budget.Choice := [(.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 11), .second, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 29), (.firstFactor 17), .possible, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 13), .second, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 31), (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 17), .first, .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .possible, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .possible, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9283_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9283 row9283_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9283_block_007
  · decide
  · rfl

def row9283_block_008 : List Nat := [15541, 16141, 16447]
def plan9283_block_008 : List Budget.Choice := [(.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 13), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 29), (.secondFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 31), .first, .first, .possible, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 11), (.secondFactor 31), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .possible, .first, .second, (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9283_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9283 row9283_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9283_block_008
  · decide
  · rfl

def row9283_block_009 : List Nat := []
def plan9283_block_009 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 17), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 29), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .second, (.firstFactor 11), .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9283_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9283 row9283_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9283_block_009
  · decide
  · rfl

def row9283_block_010 : List Nat := []
theorem row9283_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9283 row9283_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9283_block_011 : List Nat := []
theorem row9283_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9283 row9283_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9283_block_012 : List Nat := []
theorem row9283_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9283 row9283_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9283_block_013 : List Nat := []
theorem row9283_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9283 row9283_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9283_large_left_left : List Nat := row9283_small ++ row9283_block_005
theorem row9283_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9283 row9283_large_left_left :=
  Budget.RowUpperBound.branch row9283_small_covers row9283_block_005_covers

def row9283_large_left_right_right : List Nat := row9283_block_007 ++ row9283_block_008
theorem row9283_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9283 row9283_large_left_right_right :=
  Budget.RowUpperBound.branch row9283_block_007_covers row9283_block_008_covers

def row9283_large_left_right : List Nat := row9283_block_006 ++ row9283_large_left_right_right
theorem row9283_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9283 row9283_large_left_right :=
  Budget.RowUpperBound.branch row9283_block_006_covers row9283_large_left_right_right_covers

def row9283_large_left : List Nat := row9283_large_left_left ++ row9283_large_left_right
theorem row9283_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9283 row9283_large_left :=
  Budget.RowUpperBound.branch row9283_large_left_left_covers row9283_large_left_right_covers

def row9283_large_right_left : List Nat := row9283_block_009 ++ row9283_block_010
theorem row9283_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9283 row9283_large_right_left :=
  Budget.RowUpperBound.branch row9283_block_009_covers row9283_block_010_covers

def row9283_large_right_right_right : List Nat := row9283_block_012 ++ row9283_block_013
theorem row9283_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9283 row9283_large_right_right_right :=
  Budget.RowUpperBound.branch row9283_block_012_covers row9283_block_013_covers

def row9283_large_right_right : List Nat := row9283_block_011 ++ row9283_large_right_right_right
theorem row9283_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9283 row9283_large_right_right :=
  Budget.RowUpperBound.branch row9283_block_011_covers row9283_large_right_right_right_covers

def row9283_large_right : List Nat := row9283_large_right_left ++ row9283_large_right_right
theorem row9283_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9283 row9283_large_right :=
  Budget.RowUpperBound.branch row9283_large_right_left_covers row9283_large_right_right_covers

def row9283_large : List Nat := row9283_large_left ++ row9283_large_right
theorem row9283_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9283 row9283_large :=
  Budget.RowUpperBound.branch row9283_large_left_covers row9283_large_right_covers

def row9283 : List Nat := row9283_large
theorem row9283_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9283 row9283 := row9283_large_covers

end ProjectEulerSolutions.PrimePairLarge
