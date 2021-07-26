#include "ThreadSyncObject.h"

ThreadSyncObject::ThreadSyncObject(unsigned int numOfThreads)
    : m_NumberOfThreads(numOfThreads)
{
}

void ThreadSyncObject::Sync()
{
    if (m_NumberOfThreads <= 1)
    {
        // No synchronization required.
        return;
    }

    // construct a local lock which will unlock mutex after scope expires.
    std::unique_lock<std::mutex> local_ulock(m_Mutex);

    // lambda function to test for wait condition.
    auto conditionTestFunction = [this] {
        return (m_WaitCounter.load() == 0);
    };

    // update the counter which tells how many threads are waiting.
    // Once all threads arrive, we release the wait condition.
    m_WaitCounter.store(m_WaitCounter.load() + 1);

    if (m_WaitCounter.load() >= m_NumberOfThreads)
    {
        // reset thread counter and release all threads from wait condition.
        m_WaitCounter = 0;
        local_ulock.unlock();
        m_ConditionVariable.notify_all();
    }
    else
    {
        // wait for other threads to arrive at this point.
        m_ConditionVariable.wait(local_ulock, conditionTestFunction);
    }
}