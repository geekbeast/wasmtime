;;! target = "x86_64"

(module
  (func (export "singular") (result i32)
    (loop (nop))
    (loop (result i32) (i32.const 7))
  )
)
;;    0:	 55                   	push	rbp
;;    1:	 4889e5               	mov	rbp, rsp
;;    4:	 4883ec08             	sub	rsp, 8
;;    8:	 4c893424             	mov	qword ptr [rsp], r14
;;    c:	 48c7c007000000       	mov	rax, 7
;;   13:	 4883c408             	add	rsp, 8
;;   17:	 5d                   	pop	rbp
;;   18:	 c3                   	ret	
