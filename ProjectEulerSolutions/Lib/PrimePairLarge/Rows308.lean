import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10067_small : List Nat := []
theorem row10067_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10067 row10067_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10067_block_005 : List Nat := []
def plan10067_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .second, .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10067_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10067 row10067_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10067_block_005
  · decide
  · rfl

def row10067_block_006 : List Nat := [11003, 11801, 11807]
def plan10067_block_006 : List Budget.Choice := [(.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 17), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .possible, .possible, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 31), .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 37), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10067_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10067 row10067_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10067_block_006
  · decide
  · rfl

def row10067_block_007 : List Nat := [13121, 13907]
def plan10067_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), .second, .first, .first, .first, .first, .second, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 19), (.secondFactor 17), .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 17), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10067_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10067 row10067_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10067_block_007
  · decide
  · rfl

def row10067_block_008 : List Nat := [15629, 15683]
def plan10067_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, .second, .first, (.firstFactor 37), (.secondFactor 7), (.secondFactor 13), .first, .first, .second, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 37), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 11), (.secondFactor 23), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), (.secondFactor 37), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .possible, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), .first, .second, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10067_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10067 row10067_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10067_block_008
  · decide
  · rfl

def row10067_block_009 : List Nat := []
theorem row10067_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10067 row10067_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10067_block_010 : List Nat := []
theorem row10067_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10067 row10067_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10067_block_011 : List Nat := []
theorem row10067_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10067 row10067_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10067_block_012 : List Nat := []
theorem row10067_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10067 row10067_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10067_block_013 : List Nat := []
theorem row10067_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10067 row10067_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10067_large_left_left : List Nat := row10067_small ++ row10067_block_005
theorem row10067_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10067 row10067_large_left_left :=
  Budget.RowUpperBound.branch row10067_small_covers row10067_block_005_covers

def row10067_large_left_right_right : List Nat := row10067_block_007 ++ row10067_block_008
theorem row10067_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10067 row10067_large_left_right_right :=
  Budget.RowUpperBound.branch row10067_block_007_covers row10067_block_008_covers

def row10067_large_left_right : List Nat := row10067_block_006 ++ row10067_large_left_right_right
theorem row10067_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10067 row10067_large_left_right :=
  Budget.RowUpperBound.branch row10067_block_006_covers row10067_large_left_right_right_covers

def row10067_large_left : List Nat := row10067_large_left_left ++ row10067_large_left_right
theorem row10067_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10067 row10067_large_left :=
  Budget.RowUpperBound.branch row10067_large_left_left_covers row10067_large_left_right_covers

def row10067_large_right_left : List Nat := row10067_block_009 ++ row10067_block_010
theorem row10067_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10067 row10067_large_right_left :=
  Budget.RowUpperBound.branch row10067_block_009_covers row10067_block_010_covers

def row10067_large_right_right_right : List Nat := row10067_block_012 ++ row10067_block_013
theorem row10067_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10067 row10067_large_right_right_right :=
  Budget.RowUpperBound.branch row10067_block_012_covers row10067_block_013_covers

def row10067_large_right_right : List Nat := row10067_block_011 ++ row10067_large_right_right_right
theorem row10067_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10067 row10067_large_right_right :=
  Budget.RowUpperBound.branch row10067_block_011_covers row10067_large_right_right_right_covers

def row10067_large_right : List Nat := row10067_large_right_left ++ row10067_large_right_right
theorem row10067_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10067 row10067_large_right :=
  Budget.RowUpperBound.branch row10067_large_right_left_covers row10067_large_right_right_covers

def row10067_large : List Nat := row10067_large_left ++ row10067_large_right
theorem row10067_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10067 row10067_large :=
  Budget.RowUpperBound.branch row10067_large_left_covers row10067_large_right_covers

def row10067 : List Nat := row10067_large
theorem row10067_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10067 row10067 := row10067_large_covers

def row10069_small : List Nat := []
theorem row10069_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10069 row10069_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10069_block_005 : List Nat := []
def plan10069_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row10069_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10069 row10069_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10069_block_005
  · decide
  · rfl

def row10069_block_006 : List Nat := [11047, 11491, 11839, 12301]
def plan10069_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .second, .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 37), .first, .second, .possible, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10069_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10069 row10069_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10069_block_006
  · decide
  · rfl

def row10069_block_007 : List Nat := [12637, 12979, 13093, 13399, 13591, 14071]
def plan10069_block_007 : List Budget.Choice := [.first, (.secondFactor 29), .first, (.firstFactor 29), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .second, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .possible, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .possible, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, .second, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10069_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10069 row10069_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10069_block_007
  · decide
  · rfl

def row10069_block_008 : List Nat := [14653, 14767]
def plan10069_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 13), .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.secondFactor 31), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 7), .second, (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 37), (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 29), (.firstFactor 23), .first, (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10069_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10069 row10069_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10069_block_008
  · decide
  · rfl

def row10069_block_009 : List Nat := []
theorem row10069_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10069 row10069_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10069_block_010 : List Nat := []
theorem row10069_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10069 row10069_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10069_block_011 : List Nat := []
theorem row10069_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10069 row10069_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10069_block_012 : List Nat := []
theorem row10069_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10069 row10069_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10069_block_013 : List Nat := []
theorem row10069_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10069 row10069_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10069_large_left_left : List Nat := row10069_small ++ row10069_block_005
theorem row10069_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10069 row10069_large_left_left :=
  Budget.RowUpperBound.branch row10069_small_covers row10069_block_005_covers

def row10069_large_left_right_right : List Nat := row10069_block_007 ++ row10069_block_008
theorem row10069_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10069 row10069_large_left_right_right :=
  Budget.RowUpperBound.branch row10069_block_007_covers row10069_block_008_covers

def row10069_large_left_right : List Nat := row10069_block_006 ++ row10069_large_left_right_right
theorem row10069_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10069 row10069_large_left_right :=
  Budget.RowUpperBound.branch row10069_block_006_covers row10069_large_left_right_right_covers

def row10069_large_left : List Nat := row10069_large_left_left ++ row10069_large_left_right
theorem row10069_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10069 row10069_large_left :=
  Budget.RowUpperBound.branch row10069_large_left_left_covers row10069_large_left_right_covers

def row10069_large_right_left : List Nat := row10069_block_009 ++ row10069_block_010
theorem row10069_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10069 row10069_large_right_left :=
  Budget.RowUpperBound.branch row10069_block_009_covers row10069_block_010_covers

def row10069_large_right_right_right : List Nat := row10069_block_012 ++ row10069_block_013
theorem row10069_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10069 row10069_large_right_right_right :=
  Budget.RowUpperBound.branch row10069_block_012_covers row10069_block_013_covers

def row10069_large_right_right : List Nat := row10069_block_011 ++ row10069_large_right_right_right
theorem row10069_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10069 row10069_large_right_right :=
  Budget.RowUpperBound.branch row10069_block_011_covers row10069_large_right_right_right_covers

def row10069_large_right : List Nat := row10069_large_right_left ++ row10069_large_right_right
theorem row10069_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10069 row10069_large_right :=
  Budget.RowUpperBound.branch row10069_large_right_left_covers row10069_large_right_right_covers

def row10069_large : List Nat := row10069_large_left ++ row10069_large_right
theorem row10069_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10069 row10069_large :=
  Budget.RowUpperBound.branch row10069_large_left_covers row10069_large_right_covers

def row10069 : List Nat := row10069_large
theorem row10069_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10069 row10069 := row10069_large_covers

def row10079_small : List Nat := []
theorem row10079_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10079 row10079_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10079_block_005 : List Nat := [10103]
def plan10079_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .second, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .second, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row10079_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10079 row10079_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10079_block_005
  · decide
  · rfl

def row10079_block_006 : List Nat := [10847, 11909, 11939, 12041, 12203]
def plan10079_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 13), .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), (.secondFactor 7), .possible, .first, (.firstFactor 11), (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 13), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 13), (.secondFactor 19), (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 29), .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row10079_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10079 row10079_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10079_block_006
  · decide
  · rfl

def row10079_block_007 : List Nat := [13859, 14177, 14321]
def plan10079_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 31), .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), .possible, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first]
set_option maxRecDepth 2048 in
theorem row10079_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10079 row10079_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10079_block_007
  · decide
  · rfl

def row10079_block_008 : List Nat := [14657, 14951]
def plan10079_block_008 : List Budget.Choice := [.first, (.firstFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 23), (.secondFactor 7), .first, .second, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10079_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10079 row10079_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10079_block_008
  · decide
  · rfl

def row10079_block_009 : List Nat := []
theorem row10079_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10079 row10079_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10079_block_010 : List Nat := []
theorem row10079_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10079 row10079_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10079_block_011 : List Nat := []
theorem row10079_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10079 row10079_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10079_block_012 : List Nat := []
theorem row10079_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10079 row10079_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10079_block_013 : List Nat := []
theorem row10079_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10079 row10079_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10079_large_left_left : List Nat := row10079_small ++ row10079_block_005
theorem row10079_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10079 row10079_large_left_left :=
  Budget.RowUpperBound.branch row10079_small_covers row10079_block_005_covers

def row10079_large_left_right_right : List Nat := row10079_block_007 ++ row10079_block_008
theorem row10079_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10079 row10079_large_left_right_right :=
  Budget.RowUpperBound.branch row10079_block_007_covers row10079_block_008_covers

def row10079_large_left_right : List Nat := row10079_block_006 ++ row10079_large_left_right_right
theorem row10079_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10079 row10079_large_left_right :=
  Budget.RowUpperBound.branch row10079_block_006_covers row10079_large_left_right_right_covers

def row10079_large_left : List Nat := row10079_large_left_left ++ row10079_large_left_right
theorem row10079_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10079 row10079_large_left :=
  Budget.RowUpperBound.branch row10079_large_left_left_covers row10079_large_left_right_covers

def row10079_large_right_left : List Nat := row10079_block_009 ++ row10079_block_010
theorem row10079_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10079 row10079_large_right_left :=
  Budget.RowUpperBound.branch row10079_block_009_covers row10079_block_010_covers

def row10079_large_right_right_right : List Nat := row10079_block_012 ++ row10079_block_013
theorem row10079_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10079 row10079_large_right_right_right :=
  Budget.RowUpperBound.branch row10079_block_012_covers row10079_block_013_covers

def row10079_large_right_right : List Nat := row10079_block_011 ++ row10079_large_right_right_right
theorem row10079_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10079 row10079_large_right_right :=
  Budget.RowUpperBound.branch row10079_block_011_covers row10079_large_right_right_right_covers

def row10079_large_right : List Nat := row10079_large_right_left ++ row10079_large_right_right
theorem row10079_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10079 row10079_large_right :=
  Budget.RowUpperBound.branch row10079_large_right_left_covers row10079_large_right_right_covers

def row10079_large : List Nat := row10079_large_left ++ row10079_large_right
theorem row10079_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10079 row10079_large :=
  Budget.RowUpperBound.branch row10079_large_left_covers row10079_large_right_covers

def row10079 : List Nat := row10079_large
theorem row10079_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10079 row10079 := row10079_large_covers

def row10091_small : List Nat := []
theorem row10091_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10091 row10091_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10091_block_005 : List Nat := [10133]
def plan10091_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10091_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10091 row10091_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10091_block_005
  · decide
  · rfl

def row10091_block_006 : List Nat := [10589, 10601, 10979, 12323]
def plan10091_block_006 : List Budget.Choice := [(.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 17), (.firstFactor 37), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .second, (.secondFactor 37), (.firstFactor 13), .first, (.firstFactor 29), .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .second, .first, .first, (.secondFactor 7), (.firstFactor 13), .second, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .possible, (.secondFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10091_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10091 row10091_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10091_block_006
  · decide
  · rfl

def row10091_block_007 : List Nat := [13151, 13577, 13877]
def plan10091_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 13), .first, .first, .second, .second, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 23), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, .first, .first, .second, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 29), (.firstFactor 13), .first, .possible, .first, (.firstFactor 19), (.secondFactor 19), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .second, .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 13), .second, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row10091_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10091 row10091_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10091_block_007
  · decide
  · rfl

def row10091_block_008 : List Nat := [15473]
def plan10091_block_008 : List Budget.Choice := [.first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 17), .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .second, (.firstFactor 17), (.secondFactor 23), (.firstFactor 31), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .second, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 37), .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10091_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10091 row10091_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10091_block_008
  · decide
  · rfl

def row10091_block_009 : List Nat := []
theorem row10091_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10091 row10091_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10091_block_010 : List Nat := []
theorem row10091_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10091 row10091_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10091_block_011 : List Nat := []
theorem row10091_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10091 row10091_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10091_block_012 : List Nat := []
theorem row10091_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10091 row10091_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10091_block_013 : List Nat := []
theorem row10091_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10091 row10091_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10091_large_left_left : List Nat := row10091_small ++ row10091_block_005
theorem row10091_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10091 row10091_large_left_left :=
  Budget.RowUpperBound.branch row10091_small_covers row10091_block_005_covers

def row10091_large_left_right_right : List Nat := row10091_block_007 ++ row10091_block_008
theorem row10091_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10091 row10091_large_left_right_right :=
  Budget.RowUpperBound.branch row10091_block_007_covers row10091_block_008_covers

def row10091_large_left_right : List Nat := row10091_block_006 ++ row10091_large_left_right_right
theorem row10091_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10091 row10091_large_left_right :=
  Budget.RowUpperBound.branch row10091_block_006_covers row10091_large_left_right_right_covers

def row10091_large_left : List Nat := row10091_large_left_left ++ row10091_large_left_right
theorem row10091_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10091 row10091_large_left :=
  Budget.RowUpperBound.branch row10091_large_left_left_covers row10091_large_left_right_covers

def row10091_large_right_left : List Nat := row10091_block_009 ++ row10091_block_010
theorem row10091_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10091 row10091_large_right_left :=
  Budget.RowUpperBound.branch row10091_block_009_covers row10091_block_010_covers

def row10091_large_right_right_right : List Nat := row10091_block_012 ++ row10091_block_013
theorem row10091_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10091 row10091_large_right_right_right :=
  Budget.RowUpperBound.branch row10091_block_012_covers row10091_block_013_covers

def row10091_large_right_right : List Nat := row10091_block_011 ++ row10091_large_right_right_right
theorem row10091_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10091 row10091_large_right_right :=
  Budget.RowUpperBound.branch row10091_block_011_covers row10091_large_right_right_right_covers

def row10091_large_right : List Nat := row10091_large_right_left ++ row10091_large_right_right
theorem row10091_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10091 row10091_large_right :=
  Budget.RowUpperBound.branch row10091_large_right_left_covers row10091_large_right_right_covers

def row10091_large : List Nat := row10091_large_left ++ row10091_large_right
theorem row10091_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10091 row10091_large :=
  Budget.RowUpperBound.branch row10091_large_left_covers row10091_large_right_covers

def row10091 : List Nat := row10091_large
theorem row10091_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10091 row10091 := row10091_large_covers

end ProjectEulerSolutions.PrimePairLarge
